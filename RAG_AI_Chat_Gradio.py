# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "faiss-cpu",
#     "gradio>=5.0.0",
#     "huggingface-hub",
#     "llama-cpp-python",
#     "numpy",
#     "paddleocr",
#     "paddlepaddle",
#     "pymupdf",
#     "sentence-transformers",
# ]
# ///
from __future__ import annotations

import os
import sys

# Prevent macOS OpenMP / fork crashes between faiss, torch, and llama.cpp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import inspect
import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

print("[Startup] Importing libraries...", flush=True)
import faiss
import gradio as gr
import numpy as np
import pymupdf
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

UNPARSED_LABEL = "Unlabeled"
OCR_RENDER_DPI = 150
MIN_EMBEDDED_CHARS = 50

_ADDRESS_RE = re.compile(
    r"(@|https?://|www\.|\.(com|org|net)\b|"
    r"\b(tel|phone|fax|e-mail|email)\b|"
    r"\b(street|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|way|lane|drive|dr\.?)\b|"
    r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b|"
    r",\s*[A-Z]{2}\s+\d{5})",
    re.I,
)
_PAGE_LINE_RE = re.compile(r"^page\s+\d+\s*(/|of)\s*\d+$", re.I)
_DATE_RE = re.compile(
    r"^(\d{1,2}[\s/-][A-Za-z]{3,9}[\s,/-]+\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})$"
)
_COMPANY_RE = re.compile(r"\b(inc|llc|ltd|corp|gmbh|ab|plc)\.?\s*$", re.I)
_SKIP_EXACT = {"united states", "usa", "u.s.a."}
_SMALL_WORDS = {"of", "and", "the", "for", "regarding", "with", "in", "on", "a", "to"}
_ROUTE_STOP = {
    "the", "and", "for", "with", "from", "that", "this",
    "what", "when", "where", "which", "page", "date", "does", "have",
}


class OcrError(RuntimeError):
    """A page image could not be transcribed by the configured OCR model."""


class OcrModelUnavailable(OcrError):
    """The configured OCR weights could not be loaded."""


class LlamaCppInstruct:
    """Chat completion against one GGUF instruct model."""

    def __init__(
        self,
        repo_id: str,
        filename: str,
        n_ctx: int = 32000,
        n_gpu_layers: int = -1,
        temperature: float = 0.0,
    ):
        self.temperature = temperature
        print(f"[1/3] Loading GGUF model {repo_id}/{filename}...", flush=True)
        try:
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=True,
            )
        except Exception as exc:
            print(f"⚠️ GPU load failed ({exc}), retrying on CPU...", flush=True)
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=0,
                n_ctx=n_ctx,
                verbose=True,
            )
        # Quiet inference logs after successful startup
        self.llm.verbose = False

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        output = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return output["choices"][0]["message"]["content"] or ""


class PaddlePageOcr:
    """Transcribe a page image with PaddleOCR (lazy-loaded on first use)."""

    def __init__(self):
        self._engine = None
        self._load_error: Optional[OcrModelUnavailable] = None

    def _ensure_loaded(self):
        if self._load_error is not None:
            raise self._load_error
        if self._engine is not None:
            return self._engine
        try:
            from paddleocr import PaddleOCR
            try:
                self._engine = PaddleOCR(lang="en")
            except TypeError:
                self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception as exc:
            self._load_error = OcrModelUnavailable(f"Could not load PaddleOCR: {exc}")
            raise self._load_error from exc
        print("PaddleOCR ready.", flush=True)
        return self._engine

    def transcribe(self, image_path: str) -> str:
        engine = self._ensure_loaded()
        try:
            if hasattr(engine, "predict"):
                raw = engine.predict(image_path)
            else:
                raw = engine.ocr(image_path, cls=True)
        except TypeError:
            raw = engine.ocr(image_path)
        except Exception as exc:
            raise OcrError(f"PaddleOCR failed on {image_path}: {exc}") from exc
        return "\n".join(_paddle_lines(raw)).strip()


def _paddle_lines(raw) -> List[str]:
    lines: List[str] = []
    if not raw:
        return lines
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        texts = getattr(item, "rec_texts", None)
        if texts is None and isinstance(item, dict):
            texts = item.get("rec_texts")
        if texts:
            lines.extend(str(text) for text in texts if text)
            continue
        if not isinstance(item, list):
            continue
        for row in item:
            if not row or not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            text_part = row[1]
            if isinstance(text_part, (list, tuple)) and text_part:
                lines.append(str(text_part[0]))
            elif isinstance(text_part, str):
                lines.append(text_part)
    return lines


@dataclass
class PageClassification:
    doc_type: str
    continues_previous: bool
    keep_page_intact: bool
    fields: Dict[str, str]


@dataclass
class PageInfo:
    page_num: int
    text: str
    doc_type: Optional[str] = None
    page_in_doc: int = 0
    fields: Dict[str, str] = field(default_factory=dict)
    keep_page_intact: bool = False
    ocr_error: Optional[str] = None


@dataclass
class LogicalDocument:
    doc_id: str
    doc_type: str
    page_start: int
    page_end: int
    text: str
    pages: List[PageInfo] = field(default_factory=list)
    fields: Dict[str, str] = field(default_factory=dict)
    chunks: Optional[List[ChunkMetadata]] = None


@dataclass
class ChunkMetadata:
    chunk_id: str
    doc_id: str
    doc_type: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    fields: Dict[str, str] = field(default_factory=dict)


def _is_letterhead(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.casefold() in _SKIP_EXACT:
        return True
    if set(stripped) <= set("-_=* "):
        return True
    if _DATE_RE.match(stripped) or _PAGE_LINE_RE.match(stripped):
        return True
    if _ADDRESS_RE.search(stripped) or _COMPANY_RE.search(stripped):
        return True
    if stripped.endswith(":") and len(stripped) <= 24:
        return True
    return False


def _headingish(line: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*", line)
    if len(words) < 2:
        return False
    good = sum(
        1 for w in words if w.isupper() or w[:1].isupper() or w.casefold() in _SMALL_WORDS
    )
    return good / len(words) >= 0.7


def _title_candidate(line: str) -> bool:
    if "docusign" in line.casefold() or re.fullmatch(r"table of contents", line.strip(), re.I):
        return False
    if line[:1].isdigit() or _is_letterhead(line) or len(line) > 110:
        return False
    if line.endswith(".") and len(line.split()) > 6:
        return False
    if not any(ch.isalpha() for ch in line):
        return False
    return _headingish(line) or (":" in line and not line.endswith(":"))


def _normalize_type(title: str) -> str:
    title = re.sub(r"\(continued\)", "", title, flags=re.I)
    title = re.sub(r"\bNo\.\s*\S+", "", title, flags=re.I)
    title = re.sub(r",?\s*Page\s+\d+\s+of\s+\d+", "", title, flags=re.I)
    return " ".join(title.split())


def _keep_field(key: str, value: str) -> bool:
    if len(key) < 4 or len(key) > 40 or not value or len(value) > 160:
        return False
    if key[0].isdigit() or not any(ch.isalpha() for ch in key):
        return False
    if "docusign" in key.casefold() or "docusign" in value.casefold():
        return False
    if _is_letterhead(key) or _is_letterhead(value) or _ADDRESS_RE.search(value):
        return False
    return True


def _header_fields(lines: List[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" in line and not line.endswith(":"):
            key, value = (part.strip() for part in line.split(":", 1))
            if _keep_field(key, value):
                fields.setdefault(key, value)
            i += 1
            continue
        if line.endswith(":") and i + 1 < len(lines):
            key = line[:-1].strip()
            value = lines[i + 1].strip()
            if value and not value.endswith(":") and _keep_field(key, value):
                fields.setdefault(key, value)
                i += 2
                continue
        i += 1
    return fields


def _without_docusign(text: str) -> str:
    return "\n".join(
        line for line in (text or "").splitlines() if "docusign" not in line.casefold()
    )


def parse_header_metadata(text: str) -> PageClassification:
    text = _without_docusign(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = lines[:18]
    title = ""
    raw = ""
    index = 0
    while index < len(header):
        line = header[index]
        if not _title_candidate(line):
            index += 1
            continue
        nxt = header[index + 1] if index + 1 < len(header) else ""
        if nxt and (_ADDRESS_RE.search(nxt) or _COMPANY_RE.search(nxt)):
            index += 1
            continue
        parts = [line]
        if (
            len(line) >= 42
            and nxt
            and ":" not in nxt
            and _title_candidate(nxt)
            and not nxt.endswith(",")
            and "whom" not in nxt.casefold()
        ):
            parts.append(nxt)
        raw = " ".join(parts)
        title = _normalize_type(raw)[:120]
        break
    return PageClassification(
        doc_type=title or UNPARSED_LABEL,
        continues_previous="continued" in raw.casefold(),
        keep_page_intact=False,
        fields=_header_fields(lines),
    )


def _clean_label(reply: str) -> str:
    if not reply or not reply.strip():
        return ""
    line = reply.strip().splitlines()[0].strip().strip("\"'`")
    for prefix in ("label:", "document:", "page:"):
        if line.casefold().startswith(prefix):
            line = line[len(prefix) :].strip()
    return " ".join(line.split())[:120]


class HeaderMetadata:
    """Classifies pages using header heuristics and the instruct LLM."""

    def __init__(self, instruct: LlamaCppInstruct):
        self.instruct = instruct

    def classify_page(
        self, text: str, page_num: int = 0, total_pages: int = 1
    ) -> PageClassification:
        found = parse_header_metadata(text)
        label = self._label_with_model(text, page_num, total_pages)
        if label:
            found.doc_type = label
        if page_num == 0:
            found.continues_previous = False
        found.keep_page_intact = len(found.fields) >= 3
        return found

    def _label_with_model(self, text: str, page_num: int, total_pages: int) -> str:
        body = _without_docusign(text).strip()
        if not body:
            return ""
        prompt = (
            "Read this entire page. Choose one short label for it.\n"
            "Use the page's own heading and what the page is about.\n"
            "Do not use DocuSign text, envelope IDs, or signature stamps.\n"
            "Return only the label.\n\n"
            f"Page {page_num + 1} of {total_pages}\n\n"
            f"{body}"
        )
        try:
            reply = self.instruct.complete(prompt, max_tokens=48)
        except Exception as exc:
            print(f"  Page {page_num + 1}: label error - {exc}", flush=True)
            return ""
        label = _clean_label(reply)
        return "" if not label or "docusign" in label.casefold() else label

    def route(self, query: str, doc_types: List[str]) -> Optional[str]:
        unique = list(dict.fromkeys(dt for dt in doc_types if dt))
        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]
        query_words = {
            w for w in re.findall(r"[a-z0-9]{3,}", query.casefold()) if w not in _ROUTE_STOP
        }
        best_type, best_score = None, 0
        for doc_type in unique:
            type_words = set(re.findall(r"[a-z0-9]{3,}", doc_type.casefold()))
            score = len(query_words & type_words)
            if score > best_score:
                best_type, best_score = doc_type, score
        return best_type if best_score else None


def extract_and_analyze_pdf(
    pdf_file,
    ocr: PaddlePageOcr,
    classifier: HeaderMetadata,
    render_dpi: int = OCR_RENDER_DPI,
) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    print("Starting PDF extraction and analysis...", flush=True)
    file_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    pdf_doc = pymupdf.open(file_path)
    try:
        total_pages = pdf_doc.page_count
        pages_info: List[PageInfo] = []
        for i, page in enumerate(pdf_doc):
            text = (page.get_text("text") or "").strip()
            ocr_error = None
            if len(text) >= MIN_EMBEDDED_CHARS:
                print(f"  Page {i + 1}: embedded text ({len(text)} chars).", flush=True)
            else:
                print(
                    f"  Page {i + 1}: {len(text)} embedded chars, running PaddleOCR...",
                    flush=True,
                )
                tmp_png = os.path.join(
                    tempfile.gettempdir(), f"ocr_page_{i}_{os.getpid()}.png"
                )
                page.get_pixmap(dpi=render_dpi).save(tmp_png)
                try:
                    text = ocr.transcribe(tmp_png).strip()
                    print(f"  Page {i + 1}: PaddleOCR extracted {len(text)} chars.", flush=True)
                except OcrModelUnavailable:
                    raise
                except Exception as exc:
                    text = ""
                    ocr_error = str(exc)
                    print(f"  Page {i + 1}: PaddleOCR failed - {exc}", flush=True)
                finally:
                    if os.path.exists(tmp_png):
                        try:
                            os.remove(tmp_png)
                        except OSError:
                            pass
            pages_info.append(PageInfo(page_num=i, text=text, ocr_error=ocr_error))
    finally:
        pdf_doc.close()

    if not pages_info:
        raise ValueError("No pages could be extracted from PDF.")

    for i, page_obj in enumerate(pages_info):
        classified = classifier.classify_page(
            page_obj.text, page_num=i, total_pages=total_pages
        )
        continued = (
            i > 0
            and classified.continues_previous
            and classified.doc_type == pages_info[i - 1].doc_type
            and classified.doc_type != UNPARSED_LABEL
        )
        if i == 0 or not continued:
            page_obj.page_in_doc = 0
            page_obj.doc_type = classified.doc_type
        else:
            page_obj.page_in_doc = pages_info[i - 1].page_in_doc + 1
            page_obj.doc_type = pages_info[i - 1].doc_type

        if "docusign" in (page_obj.doc_type or "").casefold():
            page_obj.doc_type = UNPARSED_LABEL

        page_obj.fields = {
            k: v
            for k, v in classified.fields.items()
            if "docusign" not in k.casefold() and "docusign" not in v.casefold()
        }
        page_obj.keep_page_intact = classified.keep_page_intact
        print(
            f"  Page {i + 1}: {page_obj.doc_type} (page_in_doc={page_obj.page_in_doc})",
            flush=True,
        )

    logical_docs: List[LogicalDocument] = []
    current_pages: List[PageInfo] = []

    def flush() -> None:
        if not current_pages:
            return
        opening = current_pages[0]
        merged_fields: Dict[str, str] = {}
        for p in current_pages:
            for k, v in p.fields.items():
                merged_fields.setdefault(k, v)
        body = "\n\n".join(
            _format_page_text(p, p.text)
            for p in current_pages
            if p.text.strip() or p.fields
        )
        logical_docs.append(
            LogicalDocument(
                doc_id=f"doc_{len(logical_docs)}",
                doc_type=opening.doc_type or UNPARSED_LABEL,
                page_start=opening.page_num,
                page_end=current_pages[-1].page_num,
                text=body.strip(),
                pages=list(current_pages),
                fields=merged_fields,
            )
        )

    for page in pages_info:
        if page.page_in_doc == 0 and current_pages:
            flush()
            current_pages = []
        current_pages.append(page)
    flush()
    return pages_info, logical_docs


def _format_page_text(page: PageInfo, body: str) -> str:
    lines = [f"Document type: {page.doc_type}"]
    lines.extend(f"{k}: {v}" for k, v in page.fields.items())
    prefix = "\n".join(lines)
    body = (body or "").strip()
    return f"{prefix}\n\n{body}" if body else prefix


def _append_chunk(
    chunks: List[ChunkMetadata],
    logical_doc: LogicalDocument,
    page: PageInfo,
    body: str,
) -> None:
    chunks.append(
        ChunkMetadata(
            chunk_id=f"{logical_doc.doc_id}_chunk_{len(chunks)}",
            doc_id=logical_doc.doc_id,
            doc_type=page.doc_type or logical_doc.doc_type,
            chunk_index=len(chunks),
            page_start=page.page_num,
            page_end=page.page_num,
            text=_format_page_text(page, body),
            fields=dict(page.fields),
        )
    )


def chunk_document_with_metadata(
    logical_doc: LogicalDocument,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[ChunkMetadata]:
    chunks: List[ChunkMetadata] = []
    for page in logical_doc.pages:
        body = page.text.strip()
        if not body and not page.fields:
            continue
        words = body.split()
        if page.keep_page_intact or len(words) <= chunk_size:
            _append_chunk(chunks, logical_doc, page, body)
            continue

        lines = body.splitlines()
        current_lines: List[str] = []
        current_words = 0
        for line in lines:
            line_word_count = len(line.split())
            if current_words + line_word_count > chunk_size and current_lines:
                _append_chunk(chunks, logical_doc, page, "\n".join(current_lines))
                overlap_lines: List[str] = []
                overlap_words = 0
                for prev_line in reversed(current_lines):
                    word_len = len(prev_line.split())
                    if overlap_words + word_len > overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_words += word_len
                current_lines = overlap_lines
                current_words = overlap_words
            current_lines.append(line)
            current_words += line_word_count
        if current_lines:
            _append_chunk(chunks, logical_doc, page, "\n".join(current_lines))
    return chunks


class IntelligentRetriever:
    def __init__(self, classifier: HeaderMetadata, embed_model: SentenceTransformer):
        self.classifier = classifier
        self.embed_model = embed_model
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks_metadata: List[ChunkMetadata] = []
        self.doc_type_indices: Dict[str, dict] = {}

    def build_indices(self, chunks_metadata: List[ChunkMetadata]) -> None:
        self.chunks_metadata = chunks_metadata
        self.doc_type_indices = {}
        if not chunks_metadata:
            raise ValueError("OCR produced no indexable text.")

        texts = [chunk.text for chunk in chunks_metadata]
        embeddings = np.array(
            self.embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True),
            dtype="float32",
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        for doc_type in dict.fromkeys(chunk.doc_type for chunk in chunks_metadata):
            type_indices = [
                i for i, chunk in enumerate(chunks_metadata) if chunk.doc_type == doc_type
            ]
            type_index = faiss.IndexFlatIP(dim)
            type_index.add(embeddings[type_indices])
            self.doc_type_indices[doc_type] = {"index": type_index, "mapping": type_indices}

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter_doc_type: Optional[str] = None,
        auto_route: bool = True,
    ) -> Tuple[List[Tuple[ChunkMetadata, float]], str]:
        if filter_doc_type and filter_doc_type != "All":
            target_type = filter_doc_type
            routed_label = f"Filter: {target_type}"
        elif auto_route:
            target_type = self.classifier.route(query, list(self.doc_type_indices))
            routed_label = f"Auto-Routed: {target_type}" if target_type else "All Documents"
        else:
            target_type = None
            routed_label = "All Documents"

        query_embedding = np.array(
            self.embed_model.encode([query], normalize_embeddings=True),
            dtype="float32",
        )

        if target_type and target_type in self.doc_type_indices:
            type_data = self.doc_type_indices[target_type]
            search_k = min(k, len(type_data["mapping"]))
            distances, indices = type_data["index"].search(query_embedding, search_k)
            pairs = [
                (
                    self.chunks_metadata[type_data["mapping"][i]],
                    float(max(0.0, distances[0][idx])),
                )
                for idx, i in enumerate(indices[0])
                if i != -1
            ]
            pairs.sort(key=lambda item: (item[0].page_start, item[0].chunk_index))
            return pairs, routed_label

        search_k = min(k, len(self.chunks_metadata))
        distances, indices = self.index.search(query_embedding, search_k)
        return (
            [
                (self.chunks_metadata[i], float(max(0.0, distances[0][idx])))
                for idx, i in enumerate(indices[0])
                if i != -1
            ],
            routed_label,
        )


class EnhancedDocumentStore:
    def __init__(
        self,
        ocr: PaddlePageOcr,
        classifier: HeaderMetadata,
        embed_model: SentenceTransformer,
        instruct: LlamaCppInstruct,
    ):
        self.ocr = ocr
        self.classifier = classifier
        self.instruct = instruct
        self.retriever = IntelligentRetriever(classifier, embed_model)
        self.pages_info: List[PageInfo] = []
        self.logical_docs: List[LogicalDocument] = []
        self.chunks_metadata: List[ChunkMetadata] = []
        self.is_ready = False
        self.processing_stats: dict = {}
        self.last_results_dict: dict = {}

    def process_pdf(self, pdf_file, filename: str = "document.pdf") -> Tuple[bool, dict]:
        self.is_ready = False
        start_time = time.perf_counter()
        try:
            self.pages_info, self.logical_docs = extract_and_analyze_pdf(
                pdf_file, self.ocr, self.classifier
            )
            self.chunks_metadata = []
            for doc in self.logical_docs:
                doc.chunks = chunk_document_with_metadata(doc)
                self.chunks_metadata.extend(doc.chunks)

            self.retriever.build_indices(self.chunks_metadata)
            process_time = time.perf_counter() - start_time
            unique_types = list(dict.fromkeys(doc.doc_type for doc in self.logical_docs))
            ocr_failures = sum(1 for page in self.pages_info if page.ocr_error)
            self.processing_stats = {
                "filename": os.path.basename(str(filename)),
                "total_pages": len(self.pages_info),
                "documents_found": len(self.logical_docs),
                "total_chunks": len(self.chunks_metadata),
                "document_types": unique_types,
                "ocr_failures": ocr_failures,
                "processing_time": f"{process_time:.1f}s",
            }
            self.is_ready = True
            return True, self.processing_stats
        except Exception as exc:
            return False, {"error": str(exc)}

    @staticmethod
    def _is_answer_not_found(answer: str) -> bool:
        if not answer or not answer.strip() or answer.strip().lower().startswith("not_found"):
            return True
        not_found_phrases = (
            "not mentioned",
            "not provided",
            "does not contain",
            "no information",
            "cannot find",
            "unable to find",
        )
        lowered = answer.strip().lower()
        return any(phrase in lowered for phrase in not_found_phrases)

    @staticmethod
    def _clean_not_found_prefix(answer: str) -> str:
        cleaned = answer.strip()
        if cleaned.upper().startswith("NOT_FOUND:"):
            return cleaned[10:].strip()
        if cleaned.upper().startswith("NOT_FOUND"):
            return cleaned[9:].strip(" :-")
        return cleaned

    def _generate_answer_from_chunks(
        self, question: str, chunks_with_scores: List[Tuple[ChunkMetadata, float]]
    ) -> str:
        retrieved_text = "\n\n---\n\n".join(
            f"[Document: {chunk.doc_type} | Page {chunk.page_start + 1}]\n{chunk.text}"
            for chunk, _ in chunks_with_scores
        )
        prompt = (
            "Based ONLY on the following text, answer the user's question accurately. "
            "Copy dates, lot numbers, and other identifiers exactly as they appear. "
            'If the answer is not in the text, begin the response with "NOT_FOUND:".\n\n'
            f"Text:\n{retrieved_text}\n\nUser Question: {question}"
        )
        return self.instruct.complete(prompt, max_tokens=512).strip()

    def query(
        self,
        question: str,
        filter_type: Optional[str] = None,
        auto_route: bool = True,
        k: int = 4,
    ) -> Dict:
        if not self.is_ready:
            return {
                "answer": "⚠️ Please upload a PDF file first.",
                "sources": [],
                "filter_used": "none",
                "response_time": 0.0,
            }

        start_time = time.perf_counter()
        retrieved, routed_label = self.retriever.retrieve(
            question, k=k, filter_doc_type=filter_type, auto_route=auto_route
        )
        answer = self._generate_answer_from_chunks(question, retrieved) if retrieved else ""

        if not retrieved or self._is_answer_not_found(answer):
            for doc in self.logical_docs:
                if not doc.chunks:
                    continue
                doc_batch = [(chunk, 1.0) for chunk in doc.chunks]
                candidate_answer = self._generate_answer_from_chunks(question, doc_batch)
                if not self._is_answer_not_found(candidate_answer):
                    answer = self._clean_not_found_prefix(candidate_answer)
                    retrieved = doc_batch
                    routed_label += f" -> searched {doc.doc_type}"
                    break

        elapsed = round(time.perf_counter() - start_time, 2)
        answer = self._clean_not_found_prefix(answer)
        unique_sources = list(
            dict.fromkeys(
                f"{chunk.doc_type} (Page {chunk.page_start + 1})" for chunk, _ in retrieved
            )
        )
        self.last_results_dict = {
            "query": question,
            "predicted_doc_type": routed_label,
            "response_time_seconds": elapsed,
            "matched_chunks": [
                {
                    "text": chunk.text,
                    "score": round(score, 4),
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "doc_type": chunk.doc_type,
                        "page_start": chunk.page_start,
                        "fields": chunk.fields,
                    },
                }
                for chunk, score in retrieved
            ],
            "answer": answer,
        }
        return {
            "answer": answer,
            "sources": unique_sources,
            "filter_used": routed_label,
            "response_time": elapsed,
        }


# Initialize models with progress output
instruct_llm = LlamaCppInstruct(
    repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    filename="Mistral-7B-Instruct-v0.3-Q5_K_S.gguf",
)
ocr_llm = PaddlePageOcr()
classifier = HeaderMetadata(instruct_llm)

print("[2/3] Loading embedding model (BAAI/bge-small-en-v1.5)...", flush=True)
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")


def new_store() -> EnhancedDocumentStore:
    return EnhancedDocumentStore(ocr_llm, classifier, embed_model, instruct_llm)


doc_store = new_store()


def process_pdf_ui(file):
    global doc_store
    if file is None:
        doc_store = new_store()
        return (
            "Waiting for PDF...",
            gr.update(choices=["All"], value="All"),
            "No document processed yet.",
        )
    file_path = file.name if hasattr(file, "name") else str(file)
    success, stats = doc_store.process_pdf(file_path, filename=file_path)
    if not success:
        return (
            f"❌ Error: {stats.get('error', 'Unknown')}",
            gr.update(choices=["All"], value="All"),
            "Error processing.",
        )

    failure_note = (
        f" {stats['ocr_failures']} page(s) had no OCR text."
        if stats.get("ocr_failures")
        else ""
    )
    status_text = (
        f"✅ Processed {stats['total_pages']} pages into {stats['documents_found']} docs "
        f"({stats['processing_time']}).{failure_note}"
    )
    lines = []
    for i, doc in enumerate(doc_store.logical_docs):
        field_text = ""
        if doc.fields:
            shown = list(doc.fields.items())[:6]
            field_text = " — " + "; ".join(f"{k}: {v}" for k, v in shown)
        lines.append(
            f"{i + 1}. **{doc.doc_type}** — Pages {doc.page_start + 1} to {doc.page_end + 1}{field_text}"
        )
    breakdown_md = "\n".join(lines) if lines else "No document text extracted."
    return (
        status_text,
        gr.update(choices=["All"] + stats["document_types"], value="All"),
        breakdown_md,
    )


def handle_chat(message, history, doc_filter, auto_route, top_k):
    history = history or []
    if not message or not message.strip():
        return history, ""
    if not doc_store.is_ready:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please upload a PDF first."},
        ], ""

    res = doc_store.query(
        message,
        filter_type=None if doc_filter == "All" else doc_filter,
        auto_route=auto_route,
        k=int(top_k),
    )
    ans = res["answer"]
    meta_parts = []
    if res.get("sources"):
        meta_parts.append(f"Sources: {', '.join(res['sources'])} ({res['filter_used']})")
    meta_parts.append(f"⏱️ {res['response_time']:.2f}s")
    ans += f"\n\n*{' | '.join(meta_parts)}*"

    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ans},
    ], ""


def export_results_json() -> str:
    out_path = os.path.join(tempfile.gettempdir(), "rag_results.json")
    payload = doc_store.last_results_dict or {
        "status": "No queries have been run yet.",
        "processing_stats": doc_store.processing_stats,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def export_chat_history(history) -> str:
    out_path = os.path.join(tempfile.gettempdir(), "chat_history.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history or [], f, indent=2, ensure_ascii=False)
    return out_path


print("[3/3] Starting Gradio interface...", flush=True)
chatbot_kwargs = {"label": "Chat History", "height": "70vh"}
if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
    chatbot_kwargs["type"] = "messages"

with gr.Blocks(title="Pharmaceutical Document Q&A") as demo:
    gr.Markdown("### 📄 Pharmaceutical Document Q&A")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(**chatbot_kwargs)
            user_query = gr.Textbox(placeholder="Ask a question...", label="Your Question")
            with gr.Row():
                send_btn = gr.Button("📤 Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Chat")
        with gr.Column(scale=1):
            pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
            status_box = gr.Textbox(
                label="Status", interactive=False, value="Waiting for PDF..."
            )
            with gr.Accordion("⚙️ Options", open=False):
                doc_filter = gr.Dropdown(choices=["All"], value="All", label="Filter")
                auto_route = gr.Checkbox(value=True, label="Auto-Route")
                top_k = gr.Dropdown(choices=[2, 4, 6, 8], value=4, label="Top K")
                detected_docs_md = gr.Markdown("No document processed yet.")
            with gr.Accordion("📥 Export", open=False):
                download_dict_btn = gr.Button("📥 Download JSON")
                download_chat_btn = gr.Button("💬 Download Chat")
                download_file_output = gr.File(label="Download File", interactive=False)

    pdf_input.change(
        fn=process_pdf_ui,
        inputs=pdf_input,
        outputs=[status_box, doc_filter, detected_docs_md],
    )
    send_btn.click(
        fn=handle_chat,
        inputs=[user_query, chatbot, doc_filter, auto_route, top_k],
        outputs=[chatbot, user_query],
    )
    user_query.submit(
        fn=handle_chat,
        inputs=[user_query, chatbot, doc_filter, auto_route, top_k],
        outputs=[chatbot, user_query],
    )
    clear_btn.click(lambda: [], outputs=chatbot)
    download_dict_btn.click(fn=export_results_json, outputs=download_file_output)
    download_chat_btn.click(
        fn=export_chat_history, inputs=chatbot, outputs=download_file_output
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)