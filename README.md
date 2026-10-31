# Pharmaceutical Document Q&A

Local retrieval-augmented Q&A for multi-document pharmaceutical PDFs. Upload a vendor packet, or import a saved FAISS store, then ask questions in Gradio. Answers come from pages in that file. The app runs on your machine: a GGUF instruct model you select, `BAAI/bge-small-en-v1.5` embeddings, FAISS, and PaddleOCR when a page has little embedded text.

The application is `RAG_AI_Chat_Gradio.py`. Evaluation lives in `eval_rag_pipeline.py`. Notebooks under `Notebooks/` are the research trail: OCR, page labeling, chunk metadata, query routing, and the Gradio interface.

## What the app does

1. **Start without an instruct model.** Gradio loads embeddings only. Pick a cached GGUF from the dropdown, or paste `owner/repo/file.gguf` to download one from Hugging Face. Chat, evaluation, and LLM page labels require a loaded instruct model. Header heuristics and OCR still run without one.
2. Read each page with PyMuPDF. If a page has fewer than 50 characters of embedded text, render it and transcribe it with PaddleOCR.
3. Label the page from its heading and content. Header-heuristic titles are capped at 15 words. If an instruct model is loaded, it also proposes a short page label. Header fields such as dates and identifiers stay with the page. DocuSign stamps are dropped.
4. Group consecutive pages that continue the same document, then split long pages into overlapping chunks (500 words, 100-word overlap). A page with several header fields is kept intact.
5. Embed chunks with `BAAI/bge-small-en-v1.5` and index them in FAISS, including one inner-product index per detected document type.
6. Answer a question from the top 4 matching chunks. Auto-route limits search to the document type whose label overlaps the question. If that search does not contain the answer, the app searches each logical document in turn.
7. Show the answer with page sources. Export the last result or the chat as JSON. Export or import the in-memory vector store as a zip (`index.faiss`, `chunks.json`, per-type indices, `manifest.json`).
8. Run the evaluation suite against a ground-truth JSON. The scorecard covers header fields, retrieval, ROUGE-1, latency, robustness, and an LLM rating (0–5) of each generated answer against `reference_answer`. A Questions table lists each item, ground truth, generated answer, and full LLM reason.

## Requirements

- Python 3.12. The repository pin is in `.python-version`. The Gradio script header accepts Python 3.10 through 3.12.
- [uv](https://docs.astral.sh/uv/) for installs and for launching the app and JupyterLab.
- Disk space for model weights. A 14B GGUF is several gigabytes. The embedding model and PaddleOCR weights are smaller and download on first use.
- A network connection the first time each model is fetched from Hugging Face. Later runs use the local cache. **Use selected model** loads from cache only (`local_files_only`).
- Optional GPU. The instruct model tries GPU offload first and falls back to CPU. Embeddings run on CPU.

No API key is required for `RAG_AI_Chat_Gradio.py`. Some earlier notebooks call Gemini and do need a Google API key. Those notebooks are separate from the standalone app.

## Install uv

macOS, with Homebrew:

```bash
brew install uv
```

macOS or Linux, official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone the repository and move into it:

```bash
git clone git@github.com:leonewt0n/Pfizer-RAG-AI-Pipeline.git
cd Pfizer-RAG-AI-Pipeline
```

## Run the Gradio app

From the repository root:

```bash
uv run RAG_AI_Chat_Gradio.py
```

`uv` reads the script header at the top of `RAG_AI_Chat_Gradio.py` and installs that script's dependencies into its own environment. This is the standalone app path. It does not use the Jupyter packages in `pyproject.toml`.

The first launch downloads the embedding model and can take several minutes. The terminal prints three stages:

1. Instruct GGUF is **not** loaded. Pick a cached model or download one in the UI.
2. Loading `BAAI/bge-small-en-v1.5`.
3. Starting Gradio.

Gradio opens a browser window. If it does not, use the local URL printed in the terminal. The default is [http://127.0.0.1:7860](http://127.0.0.1:7860). The server stays on this machine (`share` is off).

Leave the terminal open while you use the app. Stop the server with Ctrl+C. On exit the app closes the llama.cpp handle so Metal GPU buffers can release.

### Load an instruct model

1. **Cached GGUF models** lists `.gguf` files already in the Hugging Face hub cache (projector `mmproj` files are omitted). Select one and click **Use selected model**.
2. To fetch a new file, paste `owner/repo/file.gguf` (or a Hugging Face blob/resolve URL) into **Download GGUF from Hugging Face** and click **Download and use model**. Example placeholder:

   `unsloth/Ministral-3-14B-Instruct-2512-GGUF/Ministral-3-14B-Instruct-2512-UD-Q4_K_XL.gguf`

3. **Loaded model** shows the active spec. Switching models closes the previous llama.cpp handle before loading the new one. On macOS, a full app restart is safer if a swap fails.

Chat, evaluation, and LLM page labels stay blocked until a model is loaded.

### Use the chat

1. Upload a PDF with **Upload PDF**, or import a vector-store zip (below). Processing starts on upload. Status reports page count, document count, and elapsed time.
2. Ask a question in **Your Question** and press Enter or **Send**. Answers copy dates, lot numbers, and other identifiers as they appear in the text, and append sources plus response time.
3. Retrieval uses auto-route and top-k = 4. The old Options panel (filter, auto-route toggle, top-k) is gone.
4. **Clear Chat** empties the transcript. It does not drop the indexed PDF. Upload another file to replace it, or clear the file input to reset.

PaddleOCR loads the first time a page needs it. A text-based PDF can finish without loading OCR weights.

### Vector store zip

Open **Vector Store**:

- **Download vector store (.zip)** writes the in-memory FAISS index, chunk metadata, per-type indices, and a manifest (embed model name, dimension, processing stats). Process a PDF first.
- **Import vector store (.zip)** restores that zip into memory so you can chat without re-OCR. The zip index dimension must match `BAAI/bge-small-en-v1.5`. Import does not switch the instruct GGUF. Header evaluation is weaker than a fresh PDF parse because original page OCR is not in the zip.

Unsafe zip members (`..` paths) are rejected. Vector count must match `chunks.json`.

### Export

Open **Export** to download the last query as JSON, or to download the chat transcript.

### Evaluation suite

Open **Run Automated Evaluation**, optionally upload a ground-truth JSON, and click **Run Evaluation Suite**. If no file is uploaded, the runner looks for `ground_truth.json` in the project directory.

The JSON shape is:

- `pdf_path` — used only when no PDF is already indexed in the UI
- `page_header_fields` — expected header keys per 1-based page
- `rag_test_cases` — `id`, `question`, `relevant_pages`, `expected_field_substring`, `expected_numeric`, `reference_answer`

Included examples:

- `ground_truth_sample-sdf-document.json`
- `ground_truth_Redacted - Supporting Documentation - Packaging Component - Example 1.json`

Scorecard metrics:

| Metric | Meaning |
| --- | --- |
| 1a Header field exact match | Extracted header fields vs ground truth |
| 1b QA field exact match | Expected substring present in the generated answer |
| 2 Numeric mean relative error | Closest number in the answer vs `expected_numeric` |
| 3a Retrieval recall@4 | Fraction of relevant pages in the top-k |
| 3b Retrieval MRR | Reciprocal rank of the first relevant page |
| 4 ROUGE-1 F1 | Token overlap with `reference_answer` |
| 5 Avg latency | End-to-end `query` time |
| 6 Robustness error rate | Empty or not-found answers |
| 7 LLM answer rating | Instruct model scores generated vs ground truth, 0–5, facts and numbers first |

The Per-Query RAG Breakdown table is metrics only. The Questions table at the bottom has the question, ground truth, generated answer, score, and full LLM reason. The same blocks are written to a downloadable CSV.

Sample PDFs:

- `PharmaDocs/` holds vendor-style packets, including mixed blobs.
- `Claude_Generated_Samples_pharma_sdf_package_Zakaria/` holds generated lactose and magnesium stearate packets, mixed blobs, and ground-truth JSON and CSV files for checking labels and fields.

## Run JupyterLab

Notebook work uses the project environment in `pyproject.toml`, which already includes JupyterLab, ipykernel, LlamaIndex, Gradio, PaddleOCR, and `llama-cpp-python`.

From the repository root:

```bash
uv sync
uv run jupyter lab
```

`uv sync` creates `.venv` from `uv.lock` and `.python-version` (Python 3.12) if that environment is not already present. `uv run jupyter lab` starts JupyterLab inside that environment and prints a local URL with a token. Open that URL. Stop the server with Ctrl+C.

The default kernel is the project virtualenv, so notebook imports resolve against the locked dependencies. If a notebook cell runs `!uv add ...`, that changes `pyproject.toml` and `uv.lock`. Prefer `uv sync` for a clean checkout, and add a package only when you intend to keep it.

Notebooks worth opening first (all under `Notebooks/`):

| Notebook | Role |
| --- | --- |
| `gradio_RAG_chat_FINAL.ipynb` | Notebook form of the chat pipeline, before it was extracted into `RAG_AI_Chat_Gradio.py`. |
| `End_to_End_RAG_Pipeline_with_Page_Level_Metadata_and_Query_Routing.ipynb` | Page classification, logical documents, chunking, embeddings, and filtered retrieval. |
| `Full_Segmentation_Pipeline_with_Metadata_Output.ipynb` | Segmentation with structured metadata output. |
| `Route_Queries.ipynb` | Routing a question to a document type from metadata. |
| `Designing_a_Page_Level_Detection_Strategy_Using_RAG.ipynb` | Page-level detection strategy. |
| `Layout_OCR_Pharmaceutical.ipynb` | Layout-aware OCR for pharmaceutical pages. |
| `Comparing_Open_Source_Embedding_Models_for_RAG.ipynb` | Embedding model comparison. |
| `Local_LLM_Advanced_PDF_Retrieval_&_Optimization_with_LlamaIndex.ipynb` | Local LLM retrieval with LlamaIndex. |

Other notebooks (`Intro_to_Gradio.ipynb`, `Gradio_Chatbot.ipynb`, `RAG_with_Open_Source_models.ipynb`, course exercises, and earlier drafts) are kept for reference. `gradio_RAG_chat.ipynb` is an earlier chat draft. Use `gradio_RAG_chat_FINAL.ipynb` or `RAG_AI_Chat_Gradio.py` for current behavior.

## Project layout

```
RAG_AI_Chat_Gradio.py     Standalone Gradio app
eval_rag_pipeline.py      Ground-truth evaluation (UI button and CLI)
ground_truth_*.json       Example eval sets
pyproject.toml            Project dependencies, including JupyterLab
uv.lock                   Locked dependency set for uv sync
.python-version           Python 3.12
Notebooks/                Research notebooks
PharmaDocs/               Sample pharmaceutical PDFs
Claude_Generated_Samples_pharma_sdf_package_Zakaria/
                          Generated packets plus ground truth
src/pfizer_rag_ai_pipeline/
                          Package stub used by the uv project
```

`.venv/` is created locally and is gitignored.

CLI eval (uses `ground_truth.json` unless you change the loader):

```bash
uv run python eval_rag_pipeline.py
```

When the Gradio app is already running, the eval module reuses that process's `doc_store` instead of importing a second copy of the models.

## Models

| Role | Model |
| --- | --- |
| Instruct / page labels / answers / LLM-as-judge | Any GGUF you load in the UI, via `llama-cpp-python`. Placeholder: `unsloth/Ministral-3-14B-Instruct-2512-GGUF` / `Ministral-3-14B-Instruct-2512-UD-Q4_K_XL.gguf` |
| Embeddings | `BAAI/bge-small-en-v1.5` via Sentence Transformers, cosine similarity in a FAISS inner-product index |
| OCR | PaddleOCR, English, loaded on the first page that needs it |

Generation temperature is 0. Chunks default to 500 words with a 100-word overlap. Cached GGUFs are discovered with `huggingface_hub.scan_cache_dir()`.

## History

The Git history on `main` tracks the pipeline from notebooks to the standalone app:

- Initial notebooks for OCR, open-source RAG, and local Mistral routing.
- uv and JupyterLab setup, then a single simplified local LLM helper.
- Page metadata recognition, structured JSON output, and stronger page classification.
- Beta 0.2: Gradio chat works, with weak metadata labels.
- Beta 0.3: page metadata labeling improved.
- Beta 0.4: the chat pipeline extracted into `RAG_AI_Chat_Gradio.py`.
- Evaluation suite with ground-truth JSON, including an LLM rating against `reference_answer`.
- Vector-store zip export/import. Instruct GGUF is no longer loaded at startup; pick a cached file or download one.

## Notes

- Answers are limited to retrieved text. If the pages do not contain the answer, the model is instructed to begin with `NOT_FOUND:`. The UI strips that prefix.
- Page labels are produced from each page. They are not chosen from a fixed list of document types. Heuristic titles keep the first 15 words.
- On macOS, the script sets `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS=1`, and `TOKENIZERS_PARALLELISM=false` so FAISS, PyTorch, and llama.cpp can load in one process.
- `llama-cpp-python` ships wheels for common platforms. If uv has to build it from source, a C compiler and CMake must be installed.
- Hot-swapping GGUF models in one process can fail on Metal. Close-then-load is used; restart the app if a swap panics or leaves no model loaded.
- This repository processes sample and redacted pharmaceutical paperwork for document intelligence. It is not a system for regulatory submission or manufacturing release decisions.
