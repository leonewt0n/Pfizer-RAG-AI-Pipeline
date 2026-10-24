# Pharmaceutical Document Q&A

Local retrieval-augmented Q&A for multi-document pharmaceutical PDFs. Upload one PDF, including a vendor packet that mixes several document types, and ask questions in a Gradio chat. Answers come from the pages in that file. The app runs on your machine: a local Mistral instruct model, a local embedding model, and PaddleOCR when a page has little or no embedded text.

The current application is `RAG_AI_Chat_Gradio.py` (Beta 0.4). Notebooks in this repository are the research trail that produced it: OCR, page labeling, chunk metadata, query routing, and the Gradio interface.

## What the app does

1. Read each page with PyMuPDF. If a page has fewer than 50 characters of embedded text, render it and transcribe it with PaddleOCR.
2. Label the page from its own heading and content with Mistral 7B Instruct. Header fields such as dates and identifiers are kept with the page. DocuSign stamps are dropped.
3. Group consecutive pages that continue the same document, then split long pages into overlapping chunks. A page with several header fields is kept intact.
4. Embed chunks with `BAAI/bge-small-en-v1.5` and index them in FAISS, including one index per detected document type.
5. Answer a question from the top matching chunks. Auto-route can limit search to the document type whose label overlaps the question. If that search does not contain the answer, the app searches each logical document in turn.
6. Show the answer with page sources. Export the last result or the chat as JSON.

## Requirements

- Python 3.12. The repository pin is in `.python-version`. The Gradio script accepts Python 3.10 through 3.12.
- [uv](https://docs.astral.sh/uv/) for installs and for launching the app and JupyterLab.
- Disk space for model weights. The Mistral GGUF is several gigabytes. The embedding model and PaddleOCR weights are smaller and download on first use.
- A network connection the first time each model is fetched from Hugging Face. Later runs use the local cache.
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

The first launch downloads weights and can take several minutes. The terminal prints three stages:

1. Loading `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`Mistral-7B-Instruct-v0.3-Q5_K_S.gguf`). If GPU load fails, it retries on CPU.
2. Loading `BAAI/bge-small-en-v1.5`.
3. Starting Gradio.

Gradio opens a browser window. If it does not, use the local URL printed in the terminal. The default is [http://127.0.0.1:7860](http://127.0.0.1:7860). The server stays on this machine (`share` is off).

Leave the terminal open while you use the app. Stop the server with Ctrl+C.

### Use the chat

1. Upload a PDF with **Upload PDF**. Processing starts on upload. Status reports page count, document count, and elapsed time. The options panel lists each detected document and its page range.
2. Ask a question in **Your Question** and press Enter or **Send**. Answers quote dates, lot numbers, and other identifiers as they appear in the text, and append sources plus response time.
3. Open **Options** to change behavior:
   - **Filter** limits retrieval to one detected document type, or **All**.
   - **Auto-Route** (on by default) picks a document type from overlap between the question and the page labels.
   - **Top K** is 2, 4, 6, or 8 chunks. The default is 4.
4. Open **Export** to download the last query as JSON, or to download the chat transcript.
5. **Clear Chat** empties the transcript. It does not drop the indexed PDF. Upload another file to replace it, or clear the file input to reset.

PaddleOCR loads the first time a page needs it. A text-based PDF can finish without loading OCR weights.

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

Notebooks worth opening first:

| Notebook | Role |
| --- | --- |
| `gradio_RAG_chat_FINAL.ipynb` | Notebook form of the chat pipeline, before it was extracted into `RAG_AI_Chat_Gradio.py`. |
| `End_to_End_RAG_Pipeline_with_Page_Level_Metadata_and_Query_Routing.ipynb` | Page classification, logical documents, chunking, embeddings, and filtered retrieval. Contains the original uv and JupyterLab setup notes. |
| `Full_Segmentation_Pipeline_with_Metadata_Output.ipynb` | Segmentation with structured metadata output. |
| `Route_Queries.ipynb` | Routing a question to a document type from metadata. |
| `Designing_a_Page_Level_Detection_Strategy_Using_RAG.ipynb` | Page-level detection strategy. |
| `Layout_OCR_Pharmaceutical.ipynb` | Layout-aware OCR for pharmaceutical pages. |
| `Comparing_Open_Source_Embedding_Models_for_RAG.ipynb` | Embedding model comparison. |
| `Local_LLM_Advanced_PDF_Retrieval_&_Optimization_with_LlamaIndex.ipynb` | Local LLM retrieval with LlamaIndex. |

Other notebooks (`Intro_to_Gradio.ipynb`, `Gradio_Chatbot.ipynb`, `RAG_with_Open_Source_models.ipynb`, course exercises, and earlier drafts) are kept for reference. `gradio_RAG_chat.ipynb` is an earlier chat draft. Use `gradio_RAG_chat_FINAL.ipynb` or the Python file for the current behavior.

## Project layout

```
RAG_AI_Chat_Gradio.py     Standalone Gradio app (Beta 0.4)
pyproject.toml            Project dependencies, including JupyterLab
uv.lock                   Locked dependency set for uv sync
.python-version           Python 3.12
PharmaDocs/               Sample pharmaceutical PDFs
Claude_Generated_Samples_pharma_sdf_package_Zakaria/
                          Generated packets plus ground truth
src/pfizer_rag_ai_pipeline/
                          Package stub used by the uv project
```

`.venv/` is created locally and is gitignored.

## Models

| Role | Model |
| --- | --- |
| Instruct / page labels / answers | `bartowski/Mistral-7B-Instruct-v0.3-GGUF`, file `Mistral-7B-Instruct-v0.3-Q5_K_S.gguf`, via `llama-cpp-python` |
| Embeddings | `BAAI/bge-small-en-v1.5` via Sentence Transformers, cosine similarity in a FAISS inner-product index |
| OCR | PaddleOCR, English, loaded on the first page that needs it |

Generation temperature is 0. Chunks default to 500 words with a 100-word overlap.

## History

The Git history on `main` tracks the pipeline from notebooks to the standalone app:

- Initial notebooks for OCR, open-source RAG, and local Mistral routing.
- uv and JupyterLab setup, then a single simplified local LLM helper.
- Page metadata recognition, structured JSON output, and stronger page classification.
- Beta 0.2: Gradio chat works, with weak metadata labels.
- Beta 0.3: page metadata labeling improved.
- Beta 0.4: the chat pipeline extracted into `RAG_AI_Chat_Gradio.py`.

## Notes

- Answers are limited to retrieved text. If the pages do not contain the answer, the model is instructed to begin with `NOT_FOUND:`. The UI strips that prefix.
- Page labels are produced from each page. They are not chosen from a fixed list of document types.
- On macOS, the script sets `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS=1`, and `TOKENIZERS_PARALLELISM=false` so FAISS, PyTorch, and llama.cpp can load in one process.
- `llama-cpp-python` ships wheels for common platforms. If uv has to build it from source, a C compiler and CMake must be installed.
- This repository processes sample and redacted pharmaceutical paperwork for document intelligence. It is not a system for regulatory submission or manufacturing release decisions.
