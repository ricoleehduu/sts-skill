#!/usr/bin/env python3
"""
Task 3 evaluation script: CBCT dental report generation.

Usage
-----
    python score.py <input_dir> <output_dir>

Expected Codabench-style input layout
-------------------------------------
    <input_dir>/
      ref/
        Train-Labeled.csv      # or another reference CSV with Filename column
      res/
        predictions.json       # preferred

Prediction JSON format
----------------------
The preferred submission file is one JSON object mapping case id to report dict:

    {
      "100": {
        "Main appeal": "...",
        "Present medical history": "...",
        "Oral Check": "...",
        "Diagnosis": "...",
        "Treatment plan": "...",
        "Handle": "...",
        "Doctor advices": "..."
      }
    }

The script also accepts submission.json, results.json, result.json, a CSV with a
Filename/case_id/id column, or per-case .txt files as fallback.

Output
------
    <output_dir>/scores.json

The output JSON contains Weighted_Score, all mean metrics, metric summaries, case
results, missing prediction count and Overall_Status.

Metrics and weighting
---------------------
The score combines BLEU-1/2/3/4, ROUGE-L, METEOR, tooth notation precision/recall/F1,
diagnosis code recall, treatment plan recall and field completion rate using the
METRIC_WEIGHTS dictionary below.
"""
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Set

import numpy as np


CSV_COLUMNS = [
    "Filename",
    "Sex",
    "Age",
    "Main appeal",
    "Subsequent",
    "Present medical history",
    "Past medical history",
    "Oral Check",
    "Diagnosis",
    "Treatment plan",
    "Handle",
    "Doctor advices",
]

REPORT_FIELDS = [
    "Main appeal",
    "Present medical history",
    "Oral Check",
    "Diagnosis",
    "Treatment plan",
    "Handle",
    "Doctor advices",
]

METRIC_WEIGHTS = {
    "bleu_1": 0.05,
    "bleu_2": 0.05,
    "bleu_3": 0.05,
    "bleu_4": 0.05,
    "rouge_l": 0.15,
    "meteor": 0.10,
    "tooth_notation_precision": 0.10,
    "tooth_notation_recall": 0.10,
    "tooth_notation_f1": 0.10,
    "diagnosis_code_recall": 0.12,
    "treatment_plan_recall": 0.08,
    "field_completion_rate": 0.05,
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "also",
}

TREATMENT_KEYWORDS = [
    "extraction", "removal", "rct", "root canal", "filling", "restoration",
    "implant", "crown", "bridge", "scaling", "polishing", "cleaning",
    "anesthesia", "sutur", "incision", "drainage", "medication", "cement",
    "bonding", "obturation", "curettage",
]

MEDICATION_KEYWORDS = [
    "articaine", "lidocaine", "ibuprofen", "acetaminophen", "paracetamol",
    "amoxicillin", "metronidazole", "chlorhexidine", "sodium hypochlorite",
    "hydrogen peroxide", "povidone-iodine", "iodophor", "saline",
]


def clean_value(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def normalize_case_id(value) -> str:
    text = clean_value(value).strip()
    try:
        if re.fullmatch(r"\d+\.0", text):
            text = str(int(float(text)))
    except ValueError:
        pass
    return text


def find_reference_csv(ref_dir: Path) -> Path | None:
    preferred = ["Train-Labeled.csv", "Validation.csv", "Test.csv", "reference.csv", "labels.csv"]
    for name in preferred:
        matches = list(ref_dir.rglob(name))
        if matches:
            return matches[0]
    csvs = sorted(ref_dir.rglob("*.csv"))
    return csvs[0] if csvs else None


def load_reference(csv_path: Path) -> Dict[str, Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    reference = {}
    for row in rows:
        case_id = normalize_case_id(row.get("Filename") or row.get("case_id") or row.get("id"))
        if case_id:
            reference[case_id] = {k: clean_value(v) for k, v in row.items()}
    return reference


def load_predictions(res_dir: Path) -> Dict[str, Dict[str, str] | str]:
    if not res_dir.exists():
        return {}
    json_candidates = sorted(res_dir.rglob("*.json"))
    preferred = [p for p in json_candidates if p.name.lower() in {"predictions.json", "submission.json", "results.json", "result.json"}]
    if preferred or json_candidates:
        path = (preferred or json_candidates)[0]
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict) and "predictions" in data and isinstance(data["predictions"], dict):
            data = data["predictions"]
        if isinstance(data, list):
            output = {}
            for item in data:
                if isinstance(item, dict):
                    cid = normalize_case_id(item.get("Filename") or item.get("case_id") or item.get("id"))
                    if cid:
                        output[cid] = {k: clean_value(v) for k, v in item.items()}
            return output
        if isinstance(data, dict):
            return {normalize_case_id(k): v for k, v in data.items()}
        return {}

    csv_candidates = sorted(res_dir.rglob("*.csv"))
    if csv_candidates:
        with csv_candidates[0].open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        output = {}
        for row in rows:
            cid = normalize_case_id(row.get("Filename") or row.get("case_id") or row.get("id"))
            if cid:
                output[cid] = {k: clean_value(v) for k, v in row.items()}
        return output

    output = {}
    for path in sorted(res_dir.rglob("*.txt")):
        output[normalize_case_id(path.stem)] = path.read_text(encoding="utf-8", errors="ignore")
    return output


def preprocess_text(text: str) -> str:
    text = clean_value(text).lower()
    text = re.sub(r"[^\w\s\*\.#]", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    return [t for t in preprocess_text(text).split() if t not in STOPWORDS and len(t) > 1]


def ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def compute_bleu(pred: str, ref: str) -> Dict[str, float]:
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return {f"bleu_{i}": 0.0 for i in range(1, 5)}
    scores = {}
    for n in range(1, 5):
        pred_counts = Counter(ngrams(pred_tokens, n))
        ref_counts = Counter(ngrams(ref_tokens, n))
        total = sum(pred_counts.values())
        scores[f"bleu_{n}"] = float(sum((pred_counts & ref_counts).values()) / total) if total else 0.0
    return scores


def compute_rouge_l(pred: str, ref: str) -> float:
    pred_tokens = preprocess_text(pred).split()
    ref_tokens = preprocess_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
    for i, pred_token in enumerate(pred_tokens, 1):
        for j, ref_token in enumerate(ref_tokens, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if pred_token == ref_token else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def compute_meteor(pred: str, ref: str) -> float:
    pred_tokens = set(tokenize(pred))
    ref_tokens = set(tokenize(ref))
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
    recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def compute_cider(pred: str, ref: str) -> float:
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    scores = []
    for n in (1, 2, 3):
        pred_counts = Counter(ngrams(pred_tokens, n))
        ref_counts = Counter(ngrams(ref_tokens, n))
        total = sum(pred_counts.values()) + sum(ref_counts.values())
        if total:
            scores.append(sum((pred_counts & ref_counts).values()) / total)
    return float(np.mean(scores)) if scores else 0.0


def merge_report(report: Dict[str, str] | str) -> str:
    if isinstance(report, str):
        return report
    return " ".join(clean_value(report.get(field, "")).strip() for field in REPORT_FIELDS if clean_value(report.get(field, "")).strip())


def as_report_dict(report: Dict[str, str] | str) -> Dict[str, str]:
    if isinstance(report, dict):
        return {k: clean_value(v) for k, v in report.items()}
    return {"text": clean_value(report)}


def extract_tooth_notations(text: str) -> Set[str]:
    teeth = set()
    for match in re.findall(r"\*?(\d{2})", clean_value(text)):
        quadrant, tooth = int(match[0]), int(match[1])
        if 1 <= quadrant <= 4 and 1 <= tooth <= 8:
            teeth.add(f"*{match}")
    return teeth


def extract_diagnosis_codes(text: str) -> Set[str]:
    return set(re.findall(r"([A-Z]\d{2}\.?\d{0,3})", clean_value(text)))


def extract_keywords(text: str, keywords: Iterable[str]) -> Set[str]:
    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    return {m.lower() for m in pattern.findall(clean_value(text))}


def extract_entities(report: Dict[str, str]) -> dict:
    combined = " ".join(clean_value(report.get(field, "")) for field in ["Diagnosis", "Treatment plan", "Handle", "Oral Check", "Main appeal", "text"])
    return {
        "tooth_notations": extract_tooth_notations(combined),
        "diagnosis_codes": extract_diagnosis_codes(clean_value(report.get("Diagnosis", combined))),
        "treatment_actions": extract_keywords(combined, TREATMENT_KEYWORDS),
        "medications": extract_keywords(combined, MEDICATION_KEYWORDS),
    }


def precision_recall_f1(pred: Set[str], gt: Set[str]) -> tuple[float, float, float]:
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    precision = len(pred & gt) / len(pred)
    recall = len(pred & gt) / len(gt)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def compute_metrics(pred_report, gt_report) -> Dict[str, float]:
    pred_dict = as_report_dict(pred_report)
    gt_dict = as_report_dict(gt_report)
    pred_text = merge_report(pred_dict)
    gt_text = merge_report(gt_dict)
    metrics = compute_bleu(pred_text, gt_text)
    metrics["rouge_l"] = compute_rouge_l(pred_text, gt_text)
    metrics["meteor"] = compute_meteor(pred_text, gt_text)
    metrics["cider"] = compute_cider(pred_text, gt_text)

    pred_entities = extract_entities(pred_dict)
    gt_entities = extract_entities(gt_dict)
    for entity in ("tooth_notations", "diagnosis_codes", "treatment_actions", "medications"):
        precision, recall, f1 = precision_recall_f1(pred_entities[entity], gt_entities[entity])
        metrics[f"{entity}_precision"] = precision
        metrics[f"{entity}_recall"] = recall
        metrics[f"{entity}_f1"] = f1

    metrics["tooth_notation_precision"] = metrics["tooth_notations_precision"]
    metrics["tooth_notation_recall"] = metrics["tooth_notations_recall"]
    metrics["tooth_notation_f1"] = metrics["tooth_notations_f1"]
    metrics["diagnosis_code_recall"] = metrics["diagnosis_codes_recall"]
    metrics["diagnosis_exact_match"] = 1.0 if pred_entities["diagnosis_codes"] == gt_entities["diagnosis_codes"] else 0.0
    metrics["diagnosis_partial_match"] = (
        len(pred_entities["diagnosis_codes"] & gt_entities["diagnosis_codes"]) / len(gt_entities["diagnosis_codes"])
        if gt_entities["diagnosis_codes"] else 0.0
    )
    metrics["treatment_plan_recall"] = metrics["treatment_actions_recall"] if gt_entities["treatment_actions"] else 1.0

    required_fields = ["Diagnosis", "Treatment plan", "Handle"]
    pred_completed = sum(1 for field in required_fields if clean_value(pred_dict.get(field, "")).strip())
    gt_completed = sum(1 for field in required_fields if clean_value(gt_dict.get(field, "")).strip())
    metrics["pred_field_completion_rate"] = pred_completed / len(required_fields)
    metrics["gt_field_completion_rate"] = gt_completed / len(required_fields)
    metrics["field_completion_rate"] = metrics["pred_field_completion_rate"]
    return metrics


def aggregate(case_metrics: List[Dict[str, float]]) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    names = sorted({name for metrics in case_metrics for name in metrics})
    means = {}
    stds = {}
    medians = {}
    for name in names:
        values = [float(metrics.get(name, 0.0)) for metrics in case_metrics]
        means[name] = float(np.mean(values)) if values else 0.0
        stds[name] = float(np.std(values)) if values else 0.0
        medians[name] = float(median(values)) if values else 0.0
    return means, stds, medians


def weighted_score(metrics_mean: Dict[str, float]) -> float:
    total = sum(weight for name, weight in METRIC_WEIGHTS.items() if name in metrics_mean)
    if not total:
        return 0.0
    return float(sum(metrics_mean[name] * weight for name, weight in METRIC_WEIGHTS.items() if name in metrics_mean) / total)


def write_scores(output_dir: Path, scores: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4, ensure_ascii=False)


def run_evaluation(input_dir: Path, output_dir: Path) -> None:
    ref_dir = input_dir / "ref"
    res_dir = input_dir / "res"
    csv_path = find_reference_csv(ref_dir)
    if csv_path is None:
        write_scores(output_dir, {"error": f"No reference CSV found under {ref_dir}"})
        return

    reference = load_reference(csv_path)
    predictions = load_predictions(res_dir)
    case_results = []
    missing = 0
    for case_id in sorted(reference, key=lambda x: int(x) if x.isdigit() else x):
        pred = predictions.get(case_id)
        if pred is None:
            pred = {}
            missing += 1
        metrics = compute_metrics(pred, reference[case_id])
        case_results.append({"case_id": case_id, "metrics": metrics})

    metrics_mean, metrics_std, metrics_median = aggregate([r["metrics"] for r in case_results])
    score = weighted_score(metrics_mean)
    scores = {
        "Weighted_Score": score,
        "weighted_score": score,
        "Num_Cases": len(case_results),
        "Num_Missing_Predictions": missing,
        "Overall_Status": "Completed with Errors" if missing else "Success",
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
        "metrics_median": metrics_median,
        "case_results": case_results,
    }
    for name, value in metrics_mean.items():
        scores[name] = value
    write_scores(output_dir, scores)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score.py <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)
    start = time.time()
    run_evaluation(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Evaluation completed in {time.time() - start:.2f} seconds.")
