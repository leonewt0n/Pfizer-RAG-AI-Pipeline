
import csv
import json
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# Reuse the running Gradio main module if launched from the UI; otherwise import normally
rag = (
    sys.modules["__main__"]
    if hasattr(sys.modules.get("__main__"), "new_store")
    else __import__("RAG_AI_Chat_Gradio")
)

# =====================================================================
# 1. GROUND TRUTH LOADER & METRIC HELPERS
# =====================================================================


def load_ground_truth(
    gt_file: Any = None, default_path: str = "ground_truth.json"
) -> Dict[str, Any]:
    """Loads ground truth strictly from an uploaded Gradio file or local JSON file."""
    path = (
        gt_file.name
        if hasattr(gt_file, "name")
        else (str(gt_file) if gt_file else default_path)
    )
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Ground truth JSON file not found at '{path}'. "
            "Please upload a JSON file in the UI or place 'ground_truth.json' in the project directory."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_str(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[™®]", "", s)
    s = s.replace("äktat", "äkta")  # Handle PDF superscript artifact 'ÄKTAT'
    return re.sub(r"\s+", " ", s)


def compute_rouge1_f1(reference: str, candidate: str) -> float:
    ref_tokens = re.findall(r"\w+", normalize_str(reference))
    cand_tokens = re.findall(r"\w+", normalize_str(candidate))
    if not ref_tokens or not cand_tokens:
        return 0.0
    overlap = sum(1 for t in cand_tokens if t in ref_tokens)
    prec = overlap / len(cand_tokens)
    rec = overlap / len(ref_tokens)
    return round((2 * prec * rec / (prec + rec)) * 100, 2) if (prec + rec) else 0.0


def extract_numbers(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


# =====================================================================
# 2. PIPELINE EVALUATOR FOR RAG_AI_Chat_Gradio.py
# =====================================================================


def evaluate_gradio_rag(gt: Dict[str, Any], top_k: int = 2) -> Dict[str, Any]:
    # Use the active UI doc_store if a PDF is already uploaded; otherwise process from disk
    if hasattr(rag, "doc_store") and rag.doc_store.is_ready:
        store = rag.doc_store
        stats = store.processing_stats
    else:
        pdf_path = gt.get("pdf_path", "sample-sdf-document.pdf")
        if not os.path.exists(pdf_path):
            return {
                "error": (
                    f"No PDF is uploaded in the UI and '{pdf_path}' was not found on disk. "
                    "Please upload a PDF first."
                )
            }
        store = rag.new_store()
        success, stats = store.process_pdf(pdf_path, filename=pdf_path)
        if not success:
            return {"error": f"PDF processing failed: {stats}"}

    print(f"[Eval] Processed stats: {json.dumps(stats, indent=2)}")

    # -----------------------------------------------------------------
    # A. Evaluate Native Header Field Extraction (_header_fields)
    # -----------------------------------------------------------------
    header_matches = 0
    header_total = 0
    header_details = {}

    for page_key, expected_fields in gt.get("page_header_fields", {}).items():
        page_num_1idx = int(page_key)
        if page_num_1idx - 1 >= len(store.pages_info):
            continue
        page_info = store.pages_info[page_num_1idx - 1]
        extracted_fields = page_info.fields
        header_details[f"Page {page_num_1idx}"] = {}

        for k, gt_val in expected_fields.items():
            header_total += 1
            pred_val = extracted_fields.get(k, "")
            match = normalize_str(gt_val) == normalize_str(pred_val)
            header_matches += int(match)
            header_details[f"Page {page_num_1idx}"][k] = {
                "gt": gt_val,
                "pred": pred_val,
                "exact_match": match,
            }

    header_exact_match_pct = (
        round((header_matches / header_total) * 100, 2) if header_total else 0.0
    )

    # -----------------------------------------------------------------
    # B. Evaluate RAG Retrieval & Generation Across All 6 Metrics
    # -----------------------------------------------------------------
    qa_field_hits = 0
    rel_errors = []
    recalls = []
    reciprocal_ranks = []
    rouge_scores = []
    llm_scores = []
    latencies = []
    failures = 0
    query_reports = []

    for case in gt.get("rag_test_cases", []):
        qid = case["id"]
        question = case["question"]
        relevant_pages = set(case["relevant_pages"])

        # --- Metric 3: Pure Retriever Evaluation (Recall@K & MRR) ---
        retrieved_pairs, routed_label = store.retriever.retrieve(
            question, k=top_k, auto_route=True
        )
        retrieved_pages = [chunk.page_start + 1 for chunk, _ in retrieved_pairs]

        hits = len(relevant_pages.intersection(set(retrieved_pages[:top_k])))
        recall_at_k = hits / len(relevant_pages) if relevant_pages else 0.0
        recalls.append(recall_at_k)

        rr = 0.0
        for rank, p in enumerate(retrieved_pages, start=1):
            if p in relevant_pages:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # --- End-to-End Query (Metrics 1, 2, 4, 5, 6) ---
        t0 = time.perf_counter()
        try:
            res = store.query(question, auto_route=True, k=top_k)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            answer = res.get("answer", "")
            # Metric 6: Robustness check (empty answer or unhandled NOT_FOUND)
            if not answer or store._is_answer_not_found(answer):
                failures += 1

            # Metric 1 (QA Field Accuracy): Check if exact expected identifier/value is in answer
            expected_sub = normalize_str(case["expected_field_substring"])
            field_hit = expected_sub in normalize_str(answer)
            qa_field_hits += int(field_hit)

            # Metric 2 (Numeric Relative Error): Find closest extracted number to ground truth
            num_err_pct = None
            if case.get("expected_numeric") is not None:
                gt_num = float(case["expected_numeric"])
                cand_nums = extract_numbers(answer)
                if not cand_nums:
                    rel_err = 1.0
                else:
                    closest = min(cand_nums, key=lambda x: abs(x - gt_num))
                    rel_err = (
                        abs(closest - gt_num) / abs(gt_num)
                        if gt_num != 0
                        else float(closest != 0)
                    )
                rel_errors.append(rel_err)
                num_err_pct = round(rel_err * 100, 2)

            # Metric 4 (Readability - ROUGE-1 F1)
            reference_answer = case.get("reference_answer", "")
            rouge_f1 = compute_rouge1_f1(reference_answer, answer)
            rouge_scores.append(rouge_f1)

            rating = store.rate_answer_against_ground_truth(
                question, answer, reference_answer
            )
            llm_score = rating.get("score")
            llm_reason = str(rating.get("reason") or "")
            if llm_score is not None:
                llm_scores.append(llm_score)

            query_reports.append(
                {
                    "id": qid,
                    "routed_to": routed_label,
                    "retrieved_pages": retrieved_pages,
                    "relevant_pages": list(relevant_pages),
                    "recall@k": round(recall_at_k * 100, 2),
                    "rr": round(rr, 3),
                    "field_match": field_hit,
                    "numeric_rel_error_pct": num_err_pct,
                    "rouge1_f1_pct": rouge_f1,
                    "question": question,
                    "llm_score": llm_score,
                    "llm_reason": llm_reason,
                    "latency_sec": round(elapsed, 3),
                    "answer": answer,
                    "reference_answer": reference_answer,
                }
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            failures += 1
            query_reports.append({"id": qid, "question": question, "error": str(exc)})

    total_queries = len(gt.get("rag_test_cases", []))
    qa_field_acc_pct = (
        round((qa_field_hits / total_queries) * 100, 2) if total_queries else 0.0
    )

    return {
        "pipeline_processing_stats": stats,
        "summary_metrics": {
            "1a_header_field_exact_match_pct": header_exact_match_pct,
            "1b_qa_field_exact_match_pct": qa_field_acc_pct,
            "2_numeric_mean_relative_error_pct": (
                round((sum(rel_errors) / len(rel_errors)) * 100, 2)
                if rel_errors
                else 0.0
            ),
            f"3a_retrieval_recall@{top_k}_pct": (
                round((sum(recalls) / len(recalls)) * 100, 2) if recalls else 0.0
            ),
            "3b_retrieval_mrr": (
                round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
                if reciprocal_ranks
                else 0.0
            ),
            "4_readability_rouge1_f1_pct": (
                round(sum(rouge_scores) / len(rouge_scores), 2)
                if rouge_scores
                else 0.0
            ),
            "5_avg_latency_seconds": (
                round(sum(latencies) / len(latencies), 3) if latencies else 0.0
            ),
            "6_robustness_error_rate_pct": (
                round((failures / total_queries) * 100, 2) if total_queries else 0.0
            ),
            "7_llm_answer_rating_avg": (
                round(sum(llm_scores) / len(llm_scores), 2) if llm_scores else None
            ),
        },
        "header_extraction_breakdown": header_details,
        "query_breakdown": query_reports,
    }


# =====================================================================
# 3. MARKDOWN TABLE FORMATTER & CSV EXPORTER FOR GRADIO UI
# =====================================================================


def format_report_as_markdown(report: Dict[str, Any], top_k: int = 2) -> str:
    """Formats an evaluation report dictionary into Markdown tables."""
    if "error" in report:
        return f"❌ **Evaluation Error:** {report['error']}"

    top_k_int = int(top_k)
    sm = report["summary_metrics"]
    recall_key = f"3a_retrieval_recall@{top_k_int}_pct"
    llm_avg = sm.get("7_llm_answer_rating_avg")

    # Table 1: Summary Metrics Scorecard
    md = [
        "### 📊 1. Summary Metrics Scorecard",
        "| Metric | Score |",
        "| :--- | :--- |",
        f"| **1a. Header Field Accuracy (Exact Match)** | `{sm['1a_header_field_exact_match_pct']}%` |",
        f"| **1b. QA Field Accuracy (Exact Match)** | `{sm['1b_qa_field_exact_match_pct']}%` |",
        f"| **2. Numeric Relative Error (Mean)** | `{sm['2_numeric_mean_relative_error_pct']}%` |",
        f"| **3a. Retrieval Recall@{top_k_int}** | `{sm.get(recall_key, 0.0)}%` |",
        f"| **3b. Retrieval MRR** | `{sm['3b_retrieval_mrr']}` |",
        f"| **4. Readability (ROUGE-1 F1)** | `{sm['4_readability_rouge1_f1_pct']}%` |",
        f"| **5. Avg Latency** | `{sm['5_avg_latency_seconds']}s` |",
        f"| **6. Robustness (Error Rate)** | `{sm['6_robustness_error_rate_pct']}%` |",
        f"| **7. LLM Answer Rating (avg / 5)** | `{llm_avg if llm_avg is not None else 'N/A'}` |",
        "",
        "### 📑 2. Header Field Extraction",
        "| Page | Field | Ground Truth | Extracted | Match |",
        "| :--- | :--- | :--- | :--- | :---: |",
    ]

    # Table 2: Header Field Breakdown
    for page_label, fields in report.get("header_extraction_breakdown", {}).items():
        for field_name, info in fields.items():
            icon = "✅" if info["exact_match"] else "❌"
            pred_disp = str(info["pred"] or "—").replace("|", "\\|")
            gt_disp = str(info["gt"]).replace("|", "\\|")
            md.append(
                f"| {page_label} | {field_name} | `{gt_disp}` | `{pred_disp}` | {icon} |"
            )

    # Table 3: Per-Query RAG Breakdown
    md.extend(
        [
            "",
            "### 🔍 3. Per-Query RAG Breakdown",
            "| ID | Routed To | Retr. Pages | Target Pages | Recall@K | MRR | Field Match | Num Err % | ROUGE-1 | LLM /5 | Latency | Answer | LLM Reason |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |",
        ]
    )

    for q in report.get("query_breakdown", []):
        if "error" in q:
            err_msg = str(q["error"]).replace("|", "\\|")
            md.append(
                f"| **{q['id']}** | ERROR | — | — | — | — | ❌ | — | — | — | — | `{err_msg}` | — |"
            )
            continue

        f_icon = "✅" if q["field_match"] else "❌"
        num_err = (
            f"{q['numeric_rel_error_pct']}%"
            if q["numeric_rel_error_pct"] is not None
            else "N/A"
        )
        ans_clean = " ".join(str(q.get("answer", "")).split()).replace("|", "\\|")
        llm_score = q.get("llm_score")
        llm_disp = str(llm_score) if llm_score is not None else "N/A"
        reason_clean = " ".join(str(q.get("llm_reason", "")).split()).replace("|", "\\|")
        md.append(
            f"| **{q['id']}** | {q['routed_to']} | `{q['retrieved_pages']}` | `{q['relevant_pages']}` | "
            f"{q['recall@k']}% | {q['rr']} | {f_icon} | {num_err} | "
            f"{q['rouge1_f1_pct']}% | {llm_disp} | {q['latency_sec']}s | {ans_clean} | {reason_clean} |"
        )

    md.extend(
        [
            "",
            "### ❓ 4. Questions",
            "| ID | Question | Ground Truth | Generated Answer | LLM /5 | LLM Reason |",
            "| :--- | :--- | :--- | :--- | :---: | :--- |",
        ]
    )
    for q in report.get("query_breakdown", []):
        qid = q.get("id", "")
        question = " ".join(str(q.get("question", "")).split()).replace("|", "\\|")
        if "error" in q:
            err_msg = str(q["error"]).replace("|", "\\|")
            md.append(
                f"| **{qid}** | {question or '—'} | — | `{err_msg}` | — | — |"
            )
            continue
        gt_ans = " ".join(str(q.get("reference_answer", "")).split()).replace("|", "\\|")
        gen_ans = " ".join(str(q.get("answer", "")).split()).replace("|", "\\|")
        llm_score = q.get("llm_score")
        llm_disp = str(llm_score) if llm_score is not None else "N/A"
        reason = " ".join(str(q.get("llm_reason", "")).split()).replace("|", "\\|")
        md.append(
            f"| **{qid}** | {question} | {gt_ans} | {gen_ans} | {llm_disp} | {reason} |"
        )

    return "\n".join(md)


def save_report_to_csv(report: Dict[str, Any], top_k: int = 2) -> str:
    """Writes all 3 evaluation sections (Summary, Headers, Queries) to a single CSV file."""
    csv_path = os.path.join(tempfile.gettempdir(), "rag_evaluation_report.csv")
    sm = report.get("summary_metrics", {})
    recall_key = f"3a_retrieval_recall@{int(top_k)}_pct"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Section 1: Summary Metrics
        writer.writerow(["=== 1. SUMMARY METRICS SCORECARD ==="])
        writer.writerow(["Metric", "Score"])
        writer.writerow(
            [
                "1a. Header Field Accuracy (Exact Match %)",
                sm.get("1a_header_field_exact_match_pct", 0.0),
            ]
        )
        writer.writerow(
            [
                "1b. QA Field Accuracy (Exact Match %)",
                sm.get("1b_qa_field_exact_match_pct", 0.0),
            ]
        )
        writer.writerow(
            [
                "2. Numeric Relative Error (Mean %)",
                sm.get("2_numeric_mean_relative_error_pct", 0.0),
            ]
        )
        writer.writerow(
            [f"3a. Retrieval Recall@{int(top_k)} (%)", sm.get(recall_key, 0.0)]
        )
        writer.writerow(["3b. Retrieval MRR", sm.get("3b_retrieval_mrr", 0.0)])
        writer.writerow(
            [
                "4. Readability (ROUGE-1 F1 %)",
                sm.get("4_readability_rouge1_f1_pct", 0.0),
            ]
        )
        writer.writerow(
            ["5. Avg Latency (seconds)", sm.get("5_avg_latency_seconds", 0.0)]
        )
        writer.writerow(
            [
                "6. Robustness (Error Rate %)",
                sm.get("6_robustness_error_rate_pct", 0.0),
            ]
        )
        llm_avg = sm.get("7_llm_answer_rating_avg")
        writer.writerow(
            [
                "7. LLM Answer Rating (avg / 5)",
                llm_avg if llm_avg is not None else "N/A",
            ]
        )
        writer.writerow([])

        # Section 2: Header Field Extraction
        writer.writerow(["=== 2. HEADER FIELD EXTRACTION ==="])
        writer.writerow(["Page", "Field", "Ground Truth", "Extracted", "Exact Match"])
        for page_label, fields in report.get("header_extraction_breakdown", {}).items():
            for field_name, info in fields.items():
                writer.writerow(
                    [
                        page_label,
                        field_name,
                        info.get("gt", ""),
                        info.get("pred", ""),
                        "PASS" if info.get("exact_match") else "FAIL",
                    ]
                )
        writer.writerow([])

        # Section 3: Per-Query RAG Breakdown
        writer.writerow(["=== 3. PER-QUERY RAG BREAKDOWN ==="])
        writer.writerow(
            [
                "Query ID",
                "Question",
                "Routed To",
                "Retrieved Pages",
                "Target Pages",
                "Recall@K (%)",
                "MRR",
                "Field Match",
                "Numeric Rel Error (%)",
                "ROUGE-1 F1 (%)",
                "LLM Rating (/5)",
                "LLM Reason",
                "Latency (s)",
                "Generated Answer",
                "Ground Truth Answer",
            ]
        )
        for q in report.get("query_breakdown", []):
            if "error" in q:
                writer.writerow(
                    [
                        q.get("id"),
                        q.get("question", ""),
                        "ERROR",
                        "",
                        "",
                        "",
                        "",
                        "FAIL",
                        "",
                        "",
                        "",
                        "",
                        "",
                        q.get("error"),
                        "",
                    ]
                )
                continue
            writer.writerow(
                [
                    q.get("id", ""),
                    " ".join(str(q.get("question", "")).split()),
                    q.get("routed_to", ""),
                    q.get("retrieved_pages", []),
                    q.get("relevant_pages", []),
                    q.get("recall@k", 0.0),
                    q.get("rr", 0.0),
                    "PASS" if q.get("field_match") else "FAIL",
                    (
                        q.get("numeric_rel_error_pct")
                        if q.get("numeric_rel_error_pct") is not None
                        else "N/A"
                    ),
                    q.get("rouge1_f1_pct", 0.0),
                    (
                        q.get("llm_score")
                        if q.get("llm_score") is not None
                        else "N/A"
                    ),
                    q.get("llm_reason", ""),
                    q.get("latency_sec", 0.0),
                    " ".join(str(q.get("answer", "")).split()),
                    " ".join(str(q.get("reference_answer", "")).split()),
                ]
            )

        writer.writerow([])
        writer.writerow(["=== 4. QUESTIONS ==="])
        writer.writerow(
            [
                "Query ID",
                "Question",
                "Ground Truth Answer",
                "Generated Answer",
                "LLM Rating (/5)",
                "LLM Reason",
            ]
        )
        for q in report.get("query_breakdown", []):
            if "error" in q:
                writer.writerow(
                    [
                        q.get("id"),
                        q.get("question", ""),
                        "",
                        q.get("error"),
                        "",
                        "",
                    ]
                )
                continue
            writer.writerow(
                [
                    q.get("id", ""),
                    " ".join(str(q.get("question", "")).split()),
                    " ".join(str(q.get("reference_answer", "")).split()),
                    " ".join(str(q.get("answer", "")).split()),
                    (
                        q.get("llm_score")
                        if q.get("llm_score") is not None
                        else "N/A"
                    ),
                    q.get("llm_reason", ""),
                ]
            )

    return csv_path


def run_eval_as_markdown(gt_file: Any = None, top_k: int = 2) -> str:
    """Runs the evaluation and returns only the Markdown tables string."""
    md_output, _ = run_eval_with_csv(gt_file, top_k=top_k)
    return md_output


def run_eval_with_csv(
    gt_file: Any = None, top_k: int = 2
) -> Tuple[str, Optional[str]]:
    """Runs the evaluation once and returns (markdown_tables_str, csv_file_path) for Gradio."""
    try:
        gt = load_ground_truth(gt_file)
    except Exception as exc:
        return f"❌ **Could not load Ground Truth JSON:** `{exc}`", None

    top_k_int = int(top_k)
    report = evaluate_gradio_rag(gt, top_k=top_k_int)
    if "error" in report:
        return f"❌ **Evaluation Error:** {report['error']}", None

    md_output = format_report_as_markdown(report, top_k=top_k_int)
    csv_path = save_report_to_csv(report, top_k=top_k_int)
    return md_output, csv_path


if __name__ == "__main__":
    gt_data = load_ground_truth()
    report = evaluate_gradio_rag(gt_data, top_k=2)
    print("\n" + "=" * 65)
    print("📊 FINAL EVALUATION SCORECARD")
    print("=" * 65)
    print(json.dumps(report.get("summary_metrics", report), indent=2))

    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    csv_file = save_report_to_csv(report, top_k=2)
    print(f"\n✅ Saved JSON report to 'eval_report.json' and CSV report to '{csv_file}'")

