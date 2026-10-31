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

import atexit
import os
import sys

# Prevent macOS OpenMP / fork crashes between faiss, torch, and llama.cpp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import inspect
import json
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_TOP_K = 4
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_INSTRUCT_REPO = "unsloth/Ministral-3-14B-Instruct-2512-GGUF"
DEFAULT_INSTRUCT_FILENAME = "Ministral-3-14B-Instruct-2512-UD-Q4_K_XL.gguf"
DEFAULT_INSTRUCT_SPEC = f"{DEFAULT_INSTRUCT_REPO}/{DEFAULT_INSTRUCT_FILENAME}"
NO_CACHED_MODEL = "No cached GGUF models"

_HF_URL_PREFIX_RE = re.compile(r"^(?:https?://)?(?:huggingface\.co|hf\.co)/", re.I)
_LABEL_PREFIX_RE = re.compile(r"^(?:label|document|page)\s*:\s*", re.I)
_NOT_FOUND_PREFIX_RE = re.compile(r"^NOT_FOUND(?:\s*:\s*|\s*[-\s]*)", re.I)

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
_NOT_FOUND_PHRASES = (
    "not mentioned",
    "not provided",
    "does not contain",
    "no information",
    "cannot find",
    "unable to find",
)


def _resolve_file_path(file_obj: Any) -> str:
    return file_obj.name if hasattr(file_obj, "name") else str(file_obj)


def _write_json(path: str, payload: Any) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


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
        local_files_only: bool = False,
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.temperature = temperature
        print(f"Loading GGUF model {repo_id}/{filename}...", flush=True)
        load_kwargs = {
            "repo_id": repo_id,
            "filename": filename,
            "n_ctx": n_ctx,
            "verbose": True,
            "local_files_only": local_files_only,
        }
        try:
            self.llm = Llama.from_pretrained(
                n_gpu_layers=n_gpu_layers,
                flash_attn=True,
                **load_kwargs,
            )
        except Exception as exc:
            print(f"⚠️ GPU load failed ({exc}), retrying on CPU...", flush=True)
            self.llm = Llama.from_pretrained(n_gpu_layers=0, **load_kwargs)
        self.llm.verbose = False

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        output = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
        )
        return output["choices"][0]["message"]["content"] or ""


def parse_instruct_spec(raw: str) -> Tuple[str, str]:
    text = (raw or "").strip().split("?", 1)[0].strip()
    if not text:
        raise ValueError("Paste a Hugging Face GGUF path: owner/repo/file.gguf")

    text = _HF_URL_PREFIX_RE.sub("", text)
    text = text.replace("/blob/main/", "/").replace("/resolve/main/", "/")

    if not text.casefold().endswith(".gguf"):
        raise ValueError("Path must end with a .gguf filename.")

    if ":" in text:
        repo_id, filename = (part.strip() for part in text.split(":", 1))
        if repo_id and filename:
            return repo_id, filename

    parts = [part for part in text.split("/") if part]
    if len(parts) < 3:
        raise ValueError("Path must look like owner/repo/file.gguf")
    return "/".join(parts[:-1]), parts[-1]


def list_cached_gguf_specs() -> List[str]:
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir()
    except Exception as exc:
        print(f"Could not scan Hugging Face cache: {exc}", flush=True)
        return []

    specs = {
        f"{repo.repo_id}/{file_info.file_name}"
        for repo in cache.repos
        if getattr(repo, "repo_type", "model") == "model"
        for revision in repo.revisions
        for file_info in revision.files
        if str(getattr(file_info, "file_name", "")).casefold().endswith(".gguf")
        and "mmproj" not in str(getattr(file_info, "file_name", "")).casefold()
    }
    return sorted(specs)


def cached_models_dropdown(selected: Optional[str] = None):
    specs = list_cached_gguf_specs()
    if not specs:
        return gr.update(choices=[NO_CACHED_MODEL], value=NO_CACHED_MODEL)
    value = selected if selected in specs else specs[0]
    return gr.update(choices=specs, value=value)


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
            raw = (
                engine.predict(image_path)
                if hasattr(engine, "predict")
                else engine.ocr(image_path, cls=True)
            )
        except TypeError:
            raw = engine.ocr(image_path)
        except Exception as exc:
            raise OcrError(f"PaddleOCR failed on {image_path}: {exc}") from exc
        return "\n".join(_paddle_lines(raw)).strip()


def _paddle_lines(raw: Any) -> List[str]:
    if not raw:
        return []
    lines: List[str] = []
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        texts = (
            item.get("rec_texts")
            if isinstance(item, dict)
            else getattr(item, "rec_texts", None)
        )
        if texts:
            lines.extend(str(text) for text in texts if text)
            continue
        if not isinstance(item, list):
            continue
        for row in item:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
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
    return stripped.endswith(":") and len(stripped) <= 24


def _headingish(line: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'/-]*", line)
    if len(words) < 2:
        return False
    good = sum(
        1 for w in words if w[0].isupper() or w.casefold() in _SMALL_WORDS
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
    title = re.sub(r"\(continued\)|\bNo\.\s*\S+|,?\s*Page\s+\d+\s+of\s+\d+", "", title, flags=re.I)
    return " ".join(title.split()[:15])


def _keep_field(key: str, value: str) -> bool:
    if not (4 <= len(key) <= 40 and 0 < len(value) <= 160):
        return False
    if key[0].isdigit() or not any(ch.isalpha() for ch in key):
        return False
    if "docusign" in key.casefold() or "docusign" in value.casefold():
        return False
    return not (_is_letterhead(key) or _is_letterhead(value) or _ADDRESS_RE.search(value))


def _header_fields(lines: List[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" in line and not line.endswith(":"):
            key, value = (part.strip() for part in line.split(":", 1))
            if _keep_field(key, value):
                fields.setdefault(key, value)
        elif line.endswith(":") and i + 1 < len(lines):
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
    cleaned = _without_docusign(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    header = lines[:18]
    title = ""
    raw = ""

    for index, line in enumerate(header):
        if not _title_candidate(line):
            continue
        nxt = header[index + 1] if index + 1 < len(header) else ""
        if nxt and (_ADDRESS_RE.search(nxt) or _COMPANY_RE.search(nxt)):
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
        title = _normalize_type(raw)
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
    line = _LABEL_PREFIX_RE.sub("", line).strip()
    return " ".join(line.split())[:120]


def parse_llm_answer_rating(reply: str) -> Tuple[Optional[int], str]:
    text = (reply or "").strip()
    if not text:
        return None, ""
    reason_match = re.search(r"REASON\s*:\s*(.+)", text, re.I | re.S)
    reason = " ".join((reason_match.group(1) if reason_match else text).split())
    score_match = (
        re.search(r"SCORE\s*:\s*([0-5](?:\.\d+)?)\b", text, re.I)
        or re.search(r"\b([0-5](?:\.\d+)?)\s*/\s*5\b", text)
    )
    if not score_match:
        return None, reason
    score = int(round(float(score_match.group(1))))
    return max(0, min(5, score)), reason


class HeaderMetadata:
    """Classifies pages using header heuristics and the instruct LLM."""

    def __init__(self, instruct: Optional[LlamaCppInstruct] = None):
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
        if self.instruct is None:
            return ""
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
        return "" if "docusign" in label.casefold() else label

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
        return best_type


def _extract_page_info(
    page: pymupdf.Page, page_idx: int, ocr: PaddlePageOcr, render_dpi: int
) -> PageInfo:
    text = (page.get_text("text") or "").strip()
    if len(text) >= MIN_EMBEDDED_CHARS:
        print(f"  Page {page_idx + 1}: embedded text ({len(text)} chars).", flush=True)
        return PageInfo(page_num=page_idx, text=text)

    print(
        f"  Page {page_idx + 1}: {len(text)} embedded chars, running PaddleOCR...",
        flush=True,
    )
    tmp_png = os.path.join(tempfile.gettempdir(), f"ocr_page_{page_idx}_{os.getpid()}.png")
    page.get_pixmap(dpi=render_dpi).save(tmp_png)
    ocr_error = None
    try:
        text = ocr.transcribe(tmp_png).strip()
        print(f"  Page {page_idx + 1}: PaddleOCR extracted {len(text)} chars.", flush=True)
    except OcrModelUnavailable:
        raise
    except Exception as exc:
        text = ""
        ocr_error = str(exc)
        print(f"  Page {page_idx + 1}: PaddleOCR failed - {exc}", flush=True)
    finally:
        try:
            os.remove(tmp_png)
        except OSError:
            pass
    return PageInfo(page_num=page_idx, text=text, ocr_error=ocr_error)


def _build_logical_document(doc_index: int, pages: List[PageInfo]) -> LogicalDocument:
    opening = pages[0]
    merged_fields: Dict[str, str] = {}
    for p in pages:
        for k, v in p.fields.items():
            merged_fields.setdefault(k, v)
    body = "\n\n".join(
        _format_page_text(p, p.text) for p in pages if p.text.strip() or p.fields
    )
    return LogicalDocument(
        doc_id=f"doc_{doc_index}",
        doc_type=opening.doc_type or UNPARSED_LABEL,
        page_start=opening.page_num,
        page_end=pages[-1].page_num,
        text=body.strip(),
        pages=list(pages),
        fields=merged_fields,
    )


def extract_and_analyze_pdf(
    pdf_file: Any,
    ocr: PaddlePageOcr,
    classifier: HeaderMetadata,
    render_dpi: int = OCR_RENDER_DPI,
) -> Tuple[List[PageInfo], List[LogicalDocument]]:
    print("Starting PDF extraction and analysis...", flush=True)
    file_path = _resolve_file_path(pdf_file)
    with pymupdf.open(file_path) as pdf_doc:
        total_pages = pdf_doc.page_count
        pages_info = [
            _extract_page_info(page, i, ocr, render_dpi)
            for i, page in enumerate(pdf_doc)
        ]

    if not pages_info:
        raise ValueError("No pages could be extracted from PDF.")

    for i, page_obj in enumerate(pages_info):
        classified = classifier.classify_page(
            page_obj.text, page_num=i, total_pages=total_pages
        )
        prev_type = pages_info[i - 1].doc_type if i > 0 else None
        continued = (
            i > 0
            and classified.continues_previous
            and classified.doc_type == prev_type
            and classified.doc_type != UNPARSED_LABEL
        )
        if continued:
            page_obj.page_in_doc = pages_info[i - 1].page_in_doc + 1
            page_obj.doc_type = prev_type
        else:
            page_obj.page_in_doc = 0
            page_obj.doc_type = classified.doc_type

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
    for page in pages_info:
        if page.page_in_doc == 0 and current_pages:
            logical_docs.append(_build_logical_document(len(logical_docs), current_pages))
            current_pages = []
        current_pages.append(page)
    if current_pages:
        logical_docs.append(_build_logical_document(len(logical_docs), current_pages))

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

        if page.keep_page_intact or len(body.split()) <= chunk_size:
            _append_chunk(chunks, logical_doc, page, body)
            continue

        current_lines: List[Tuple[str, int]] = []
        current_words = 0
        for line in body.splitlines():
            word_count = len(line.split())
            if current_words + word_count > chunk_size and current_lines:
                _append_chunk(
                    chunks,
                    logical_doc,
                    page,
                    "\n".join(text for text, _ in current_lines),
                )
                overlap_pairs: List[Tuple[str, int]] = []
                overlap_words = 0
                for prev_line, prev_count in reversed(current_lines):
                    if overlap_words + prev_count > overlap:
                        break
                    overlap_pairs.append((prev_line, prev_count))
                    overlap_words += prev_count
                current_lines = list(reversed(overlap_pairs))
                current_words = overlap_words

            current_lines.append((line, word_count))
            current_words += word_count

        if current_lines:
            _append_chunk(
                chunks,
                logical_doc,
                page,
                "\n".join(text for text, _ in current_lines),
            )
    return chunks


def _safe_zip_name(name: str) -> str:
    cleaned = (name or "").replace("\\", "/")
    if not cleaned or cleaned.startswith("/") or ".." in cleaned.split("/"):
        raise ValueError(f"Unsafe zip member: {name}")
    return cleaned


def _chunk_from_dict(row: dict) -> ChunkMetadata:
    if not isinstance(row, dict):
        raise ValueError("Each chunk must be an object.")
    return ChunkMetadata(
        chunk_id=str(row.get("chunk_id") or ""),
        doc_id=str(row.get("doc_id") or ""),
        doc_type=str(row.get("doc_type") or UNPARSED_LABEL),
        chunk_index=int(row.get("chunk_index") or 0),
        page_start=int(row.get("page_start") or 0),
        page_end=int(row.get("page_end") or 0),
        text=str(row.get("text") or ""),
        fields=dict(row.get("fields") or {}),
    )


def _logical_docs_from_chunks(chunks: List[ChunkMetadata]) -> List[LogicalDocument]:
    grouped: Dict[str, List[ChunkMetadata]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id or "doc"].append(chunk)

    docs: List[LogicalDocument] = []
    for doc_id, doc_chunks in grouped.items():
        doc_chunks.sort(key=lambda item: (item.page_start, item.chunk_index))
        first = doc_chunks[0]
        docs.append(
            LogicalDocument(
                doc_id=doc_id,
                doc_type=first.doc_type or UNPARSED_LABEL,
                page_start=min(item.page_start for item in doc_chunks),
                page_end=max(item.page_end for item in doc_chunks),
                text="\n\n".join(item.text for item in doc_chunks),
                pages=[],
                fields=dict(first.fields),
                chunks=doc_chunks,
            )
        )
    return docs


def _pages_from_chunks(chunks: List[ChunkMetadata]) -> List[PageInfo]:
    by_page: Dict[int, List[ChunkMetadata]] = defaultdict(list)
    for chunk in chunks:
        by_page[chunk.page_start].append(chunk)

    pages: List[PageInfo] = []
    for page_num in sorted(by_page):
        group = by_page[page_num]
        fields: Dict[str, str] = {}
        for chunk in group:
            fields.update(chunk.fields)
        pages.append(
            PageInfo(
                page_num=page_num,
                text="\n\n".join(chunk.text for chunk in group),
                doc_type=group[0].doc_type,
                fields=fields,
            )
        )
    return pages


def documents_breakdown_md(store: "EnhancedDocumentStore") -> str:
    if not store.logical_docs:
        return "No document text extracted."
    lines = []
    for i, doc in enumerate(store.logical_docs, start=1):
        shown = list(doc.fields.items())[:6]
        field_text = " — " + "; ".join(f"{k}: {v}" for k, v in shown) if shown else ""
        lines.append(
            f"{i}. **{doc.doc_type}** — Pages {doc.page_start + 1} to {doc.page_end + 1}{field_text}"
        )
    return "\n".join(lines)


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
        embeddings = np.asarray(
            self.embed_model.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            ),
            dtype="float32",
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        indices_by_type: Dict[str, List[int]] = defaultdict(list)
        for i, chunk in enumerate(chunks_metadata):
            indices_by_type[chunk.doc_type].append(i)

        for doc_type, type_indices in indices_by_type.items():
            type_index = faiss.IndexFlatIP(dim)
            type_index.add(embeddings[type_indices])
            self.doc_type_indices[doc_type] = {
                "index": type_index,
                "mapping": type_indices,
            }

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

        query_embedding = np.asarray(
            self.embed_model.encode([query], normalize_embeddings=True),
            dtype="float32",
        )

        if target_type and target_type in self.doc_type_indices:
            type_data = self.doc_type_indices[target_type]
            mapping = type_data["mapping"]
            distances, indices = type_data["index"].search(
                query_embedding, min(k, len(mapping))
            )
            pairs = [
                (self.chunks_metadata[mapping[i]], float(max(0.0, distances[0][idx])))
                for idx, i in enumerate(indices[0])
                if i != -1
            ]
            pairs.sort(key=lambda item: (item[0].page_start, item[0].chunk_index))
            return pairs, routed_label

        distances, indices = self.index.search(
            query_embedding, min(k, len(self.chunks_metadata))
        )
        pairs = [
            (self.chunks_metadata[i], float(max(0.0, distances[0][idx])))
            for idx, i in enumerate(indices[0])
            if i != -1
        ]
        return pairs, routed_label


class EnhancedDocumentStore:
    def __init__(
        self,
        ocr: PaddlePageOcr,
        classifier: HeaderMetadata,
        embed_model: SentenceTransformer,
        instruct: Optional[LlamaCppInstruct] = None,
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

    def process_pdf(self, pdf_file: Any, filename: str = "document.pdf") -> Tuple[bool, dict]:
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
            self.processing_stats = {
                "filename": os.path.basename(str(filename)),
                "total_pages": len(self.pages_info),
                "documents_found": len(self.logical_docs),
                "total_chunks": len(self.chunks_metadata),
                "document_types": list(dict.fromkeys(doc.doc_type for doc in self.logical_docs)),
                "ocr_failures": sum(1 for page in self.pages_info if page.ocr_error),
                "processing_time": f"{process_time:.1f}s",
            }
            self.is_ready = True
            return True, self.processing_stats
        except Exception as exc:
            return False, {"error": str(exc)}

    def export_vector_store_zip(self) -> str:
        retriever = self.retriever
        if not self.is_ready or retriever.index is None:
            raise ValueError("Process a PDF first.")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        zip_path = os.path.join(tempfile.gettempdir(), f"vector_store_{stamp}.zip")
        with tempfile.TemporaryDirectory(prefix="rag_vector_") as work:
            index_path = os.path.join(work, "index.faiss")
            faiss.write_index(retriever.index, index_path)
            chunks_path = _write_json(
                os.path.join(work, "chunks.json"),
                [asdict(chunk) for chunk in retriever.chunks_metadata],
            )
            os.makedirs(os.path.join(work, "doc_types"), exist_ok=True)
            type_map: Dict[str, dict] = {}
            for doc_type, data in retriever.doc_type_indices.items():
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", doc_type).strip("._") or "type"
                rel = f"doc_types/{safe[:80]}.faiss"
                n = 2
                while os.path.exists(os.path.join(work, rel)):
                    rel = f"doc_types/{safe[:80]}_{n}.faiss"
                    n += 1
                faiss.write_index(data["index"], os.path.join(work, rel))
                type_map[doc_type] = {
                    "file": rel,
                    "mapping": list(data["mapping"]),
                }

            manifest = {
                "embed_model": EMBED_MODEL_NAME,
                "index_metric": "inner_product",
                "normalized_embeddings": True,
                "ntotal": int(retriever.index.ntotal),
                "dimension": int(retriever.index.d),
                "chunk_count": len(retriever.chunks_metadata),
                "processing_stats": self.processing_stats,
                "doc_type_indices": type_map,
                "instruct_model": {
                    "repo_id": getattr(self.instruct, "repo_id", DEFAULT_INSTRUCT_REPO),
                    "filename": getattr(self.instruct, "filename", DEFAULT_INSTRUCT_FILENAME),
                },
            }
            manifest_path = _write_json(os.path.join(work, "manifest.json"), manifest)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(index_path, "index.faiss")
                zf.write(chunks_path, "chunks.json")
                zf.write(manifest_path, "manifest.json")
                for info in type_map.values():
                    zf.write(os.path.join(work, info["file"]), info["file"])
        return zip_path

    def import_vector_store_zip(self, zip_file: Any) -> dict:
        zip_path = _resolve_file_path(zip_file)
        if not zip_path or not os.path.exists(zip_path):
            raise ValueError("Choose a vector store .zip file.")

        with tempfile.TemporaryDirectory(prefix="rag_vector_in_") as work:
            with zipfile.ZipFile(zip_path) as zf:
                names = {_safe_zip_name(name) for name in zf.namelist()}
                if "index.faiss" not in names or "chunks.json" not in names:
                    raise ValueError("Zip is missing index.faiss or chunks.json.")

                def extract(name: str) -> str:
                    rel = _safe_zip_name(name)
                    dest = os.path.join(work, rel)
                    os.makedirs(os.path.dirname(dest) or work, exist_ok=True)
                    Path(dest).write_bytes(zf.read(rel))
                    return dest

                index = faiss.read_index(extract("index.faiss"))
                chunks_raw = json.loads(zf.read("chunks.json").decode("utf-8"))
                if not isinstance(chunks_raw, list) or not chunks_raw:
                    raise ValueError("chunks.json is empty.")
                chunks = [_chunk_from_dict(row) for row in chunks_raw]
                if int(index.ntotal) != len(chunks):
                    raise ValueError(
                        f"Index has {index.ntotal} vectors but chunks.json has {len(chunks)}."
                    )

                embed_dim = self.retriever.embed_model.get_sentence_embedding_dimension()
                if embed_dim is None:
                    probe = self.retriever.embed_model.encode(
                        ["dimension check"], normalize_embeddings=True
                    )
                    embed_dim = int(probe.shape[-1])
                if int(index.d) != embed_dim:
                    raise ValueError(
                        f"Zip index dimension {index.d} does not match "
                        f"{EMBED_MODEL_NAME} ({embed_dim})."
                    )

                manifest = {}
                if "manifest.json" in names:
                    manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                    if not isinstance(manifest, dict):
                        raise ValueError("manifest.json must be an object.")

                type_indices: Dict[str, dict] = {}
                for doc_type, info in (manifest.get("doc_type_indices") or {}).items():
                    if not isinstance(info, dict):
                        continue
                    rel = info.get("file")
                    mapping = [int(i) for i in (info.get("mapping") or [])]
                    if not rel or _safe_zip_name(rel) not in names:
                        continue
                    type_index = faiss.read_index(extract(rel))
                    if mapping and (min(mapping) < 0 or max(mapping) >= len(chunks)):
                        raise ValueError(f"Type index mapping for {doc_type} is out of range.")
                    if int(type_index.ntotal) != len(mapping):
                        raise ValueError(
                            f"Type index {doc_type} has {type_index.ntotal} vectors "
                            f"but mapping has {len(mapping)} entries."
                        )
                    type_indices[str(doc_type)] = {
                        "index": type_index,
                        "mapping": mapping,
                    }

            self.retriever.index = index
            self.retriever.chunks_metadata = chunks
            self.retriever.doc_type_indices = type_indices
            self.chunks_metadata = chunks
            self.logical_docs = _logical_docs_from_chunks(chunks)
            self.pages_info = _pages_from_chunks(chunks)

            stats = dict(manifest.get("processing_stats") or {})
            stats.update(
                {
                    "source": os.path.basename(zip_path),
                    "total_chunks": len(chunks),
                    "documents_found": len(self.logical_docs),
                    "total_pages": len(self.pages_info),
                    "document_types": list(dict.fromkeys(chunk.doc_type for chunk in chunks)),
                }
            )
            zip_embed = str(manifest.get("embed_model") or "")
            if zip_embed and zip_embed != EMBED_MODEL_NAME:
                stats["embed_model_note"] = (
                    f"Zip listed {zip_embed}; running {EMBED_MODEL_NAME}."
                )
            self.processing_stats = stats
            self.is_ready = True
            return stats

    @staticmethod
    def _is_answer_not_found(answer: str) -> bool:
        lowered = (answer or "").strip().lower()
        if not lowered or lowered.startswith("not_found"):
            return True
        return any(phrase in lowered for phrase in _NOT_FOUND_PHRASES)

    @staticmethod
    def _clean_not_found_prefix(answer: str) -> str:
        return _NOT_FOUND_PREFIX_RE.sub("", (answer or "").strip()).strip()

    def _generate_answer_from_chunks(
        self, question: str, chunks_with_scores: List[Tuple[ChunkMetadata, float]]
    ) -> str:
        if self.instruct is None:
            return "NOT_FOUND: Load an instruct GGUF model first."
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

    def rate_answer_against_ground_truth(
        self, question: str, generated: str, ground_truth: str
    ) -> Dict:
        if self.instruct is None:
            return {"score": None, "reason": "Load an instruct GGUF model first."}
        if not (ground_truth or "").strip():
            return {"score": None, "reason": "No ground truth answer."}
        prompt = (
            "Compare the generated answer to the ground truth. Ignore style and length.\n"
            "Score 0 to 5:\n"
            "5 = all key facts match\n"
            "4 = facts match with only minor extra wording\n"
            "3 = mostly correct but missing a key fact\n"
            "2 = mixed correct and incorrect facts\n"
            "1 = mostly wrong\n"
            "0 = empty or contradicts the ground truth\n"
            "Pay special attention to numbers, dates, lot numbers, and units.\n"
            "Return exactly:\n"
            "SCORE: <integer 0-5>\n"
            "REASON: <one sentence>\n\n"
            f"Question: {question}\n"
            f"Ground truth: {ground_truth}\n"
            f"Generated: {generated}"
        )
        try:
            reply = self.instruct.complete(prompt, max_tokens=256).strip()
        except Exception as exc:
            return {"score": None, "reason": f"Rating error: {exc}"}
        score, reason = parse_llm_answer_rating(reply)
        return {"score": score, "reason": reason or "No reason returned."}

    def query(
        self,
        question: str,
        filter_type: Optional[str] = None,
        auto_route: bool = True,
        k: int = 4,
    ) -> Dict:
        if self.instruct is None:
            return {
                "answer": "⚠️ Load an instruct GGUF model first.",
                "sources": [],
                "filter_used": "none",
                "response_time": 0.0,
            }
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
                    answer = candidate_answer
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
instruct_llm: Optional[LlamaCppInstruct] = None
ocr_llm = PaddlePageOcr()
classifier = HeaderMetadata(instruct_llm)

print("[1/3] Instruct GGUF not loaded. Pick a cached model or download one.", flush=True)
print(f"[2/3] Loading embedding model ({EMBED_MODEL_NAME})...", flush=True)

embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def new_store() -> EnhancedDocumentStore:
    return EnhancedDocumentStore(ocr_llm, classifier, embed_model, instruct_llm)


doc_store = new_store()


def process_pdf_ui(file: Any):
    global doc_store
    if file is None:
        doc_store = new_store()
        return "Waiting for PDF...", "No document processed yet."
    file_path = _resolve_file_path(file)
    success, stats = doc_store.process_pdf(file_path, filename=file_path)
    if not success:
        return f"❌ Error: {stats.get('error', 'Unknown')}", "Error processing."

    failure_note = (
        f" {stats['ocr_failures']} page(s) had no OCR text."
        if stats.get("ocr_failures")
        else ""
    )
    status_text = (
        f"✅ Processed {stats['total_pages']} pages into {stats['documents_found']} docs "
        f"({stats['processing_time']}).{failure_note}"
    )
    return status_text, documents_breakdown_md(doc_store)


def handle_chat(message: str, history: Optional[list]):
    history = history or []
    if not message or not message.strip():
        return history, ""

    def _reply(text: str):
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": text},
        ], ""

    if instruct_llm is None:
        return _reply("⚠️ Load an instruct GGUF model first.")
    if not doc_store.is_ready:
        return _reply("⚠️ Upload a PDF or import a vector store first.")

    res = doc_store.query(
        message,
        filter_type=None,
        auto_route=True,
        k=DEFAULT_TOP_K,
    )
    meta_parts = []
    if res.get("sources"):
        meta_parts.append(f"Sources: {', '.join(res['sources'])} ({res['filter_used']})")
    meta_parts.append(f"⏱️ {res['response_time']:.2f}s")
    return _reply(f"{res['answer']}\n\n*{' | '.join(meta_parts)}*")


def export_results_json() -> str:
    payload = doc_store.last_results_dict or {
        "status": "No queries have been run yet.",
        "processing_stats": doc_store.processing_stats,
    }
    return _write_json(os.path.join(tempfile.gettempdir(), "rag_results.json"), payload)


def export_chat_history(history: Optional[list]) -> str:
    return _write_json(
        os.path.join(tempfile.gettempdir(), "chat_history.json"), history or []
    )


def export_vector_store_zip_ui() -> str:
    try:
        return doc_store.export_vector_store_zip()
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def import_vector_store_ui(file: Any):
    if file is None:
        return gr.update(), gr.update()
    try:
        stats = doc_store.import_vector_store_zip(file)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    note = f" {stats['embed_model_note']}" if stats.get("embed_model_note") else ""
    status_text = (
        f"✅ Imported {stats.get('total_chunks', 0)} chunks from {stats.get('source', 'zip')} "
        f"({stats.get('documents_found', 0)} docs).{note}"
    )
    return status_text, documents_breakdown_md(doc_store)


def _close_instruct_llm() -> None:
    global instruct_llm
    if instruct_llm is not None and hasattr(instruct_llm, "llm"):
        try:
            instruct_llm.llm.close()
        except Exception as exc:
            print(f"Close old model failed: {exc}", flush=True)


def load_instruct_model_ui(spec: str, local_only: bool = False):
    global instruct_llm
    try:
        repo_id, filename = parse_instruct_spec(spec)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    target_spec = f"{repo_id}/{filename}"
    current_spec = (
        f"{instruct_llm.repo_id}/{instruct_llm.filename}"
        if instruct_llm is not None
        else ""
    )
    if current_spec == target_spec:
        return f"Already using {target_spec}", cached_models_dropdown(current_spec)

    _close_instruct_llm()
    instruct_llm = None
    classifier.instruct = None
    doc_store.instruct = None
    gc.collect()

    try:
        new_model = LlamaCppInstruct(
            repo_id=repo_id,
            filename=filename,
            local_files_only=local_only,
        )
    except Exception as exc:
        raise gr.Error(f"Could not load {target_spec}: {exc}") from exc

    instruct_llm = new_model
    classifier.instruct = new_model
    doc_store.instruct = new_model
    return f"Using {target_spec}", cached_models_dropdown(target_spec)


def load_cached_model_ui(selected: str):
    if not selected or selected == NO_CACHED_MODEL:
        raise gr.Error("No cached GGUF selected.")
    return load_instruct_model_ui(selected, local_only=True)


print("[3/3] Starting Gradio interface...", flush=True)
chatbot_kwargs = {"label": "Chat History", "height": "70vh"}
if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
    chatbot_kwargs["type"] = "messages"
_startup_cached = list_cached_gguf_specs()
_startup_choices = _startup_cached or [NO_CACHED_MODEL]
_startup_selected = _startup_choices[0]

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
            cached_models = gr.Dropdown(
                label="Cached GGUF models",
                choices=_startup_choices,
                value=_startup_selected,
                interactive=True,
            )
            load_cached_btn = gr.Button("Use selected model")
            model_spec = gr.Textbox(
                label="Download GGUF from Hugging Face",
                value="",
                placeholder=DEFAULT_INSTRUCT_SPEC,
            )
            load_model_btn = gr.Button("Download and use model")
            model_status = gr.Textbox(
                label="Loaded model",
                interactive=False,
                value="No instruct model loaded",
            )
            with gr.Accordion("📦 Vector Store", open=False):
                detected_docs_md = gr.Markdown("No document processed yet.")
                import_zip_input = gr.File(
                    label="Import vector store (.zip)",
                    file_types=[".zip"],
                )
                download_index_btn = gr.Button("Download vector store (.zip)")
                vector_zip_output = gr.File(
                    label="Vector store zip", interactive=False
                )
            with gr.Accordion("📥 Export", open=False):
                download_dict_btn = gr.Button("📥 Download JSON")
                download_chat_btn = gr.Button("💬 Download Chat")
                download_file_output = gr.File(label="Download File", interactive=False)

    pdf_input.change(
        fn=process_pdf_ui,
        inputs=pdf_input,
        outputs=[status_box, detected_docs_md],
    )
    send_btn.click(
        fn=handle_chat,
        inputs=[user_query, chatbot],
        outputs=[chatbot, user_query],
    )
    user_query.submit(
        fn=handle_chat,
        inputs=[user_query, chatbot],
        outputs=[chatbot, user_query],
    )
    load_cached_btn.click(
        fn=load_cached_model_ui,
        inputs=cached_models,
        outputs=[model_status, cached_models],
    )
    load_model_btn.click(
        fn=load_instruct_model_ui,
        inputs=model_spec,
        outputs=[model_status, cached_models],
    )
    model_spec.submit(
        fn=load_instruct_model_ui,
        inputs=model_spec,
        outputs=[model_status, cached_models],
    )
    download_index_btn.click(
        fn=export_vector_store_zip_ui,
        outputs=vector_zip_output,
    )
    import_zip_input.change(
        fn=import_vector_store_ui,
        inputs=import_zip_input,
        outputs=[status_box, detected_docs_md],
    )
    clear_btn.click(lambda: [], outputs=chatbot)
    download_dict_btn.click(fn=export_results_json, outputs=download_file_output)
    download_chat_btn.click(
        fn=export_chat_history, inputs=chatbot, outputs=download_file_output
    )

    with gr.Accordion("🧪 Run Automated Evaluation", open=True):
        gt_file_input = gr.File(
            label="📋 Ground Truth JSON (optional, defaults to ground_truth.json)",
            file_types=[".json"],
        )
        eval_btn = gr.Button("▶️ Run Evaluation Suite", variant="primary")
        eval_csv_output = gr.File(
            label="📥 Download Evaluation Report (.csv)", interactive=False
        )

    eval_output_md = gr.Markdown(label="Evaluation Scorecard")
    eval_btn.click(
        fn=lambda gt_f: __import__("eval_rag_pipeline").run_eval_with_csv(
            gt_f, top_k=DEFAULT_TOP_K
        ),
        inputs=[gt_file_input],
        outputs=[eval_output_md, eval_csv_output],
    )

if __name__ == "__main__":
    atexit.register(_close_instruct_llm)
    demo.launch(inbrowser=True)