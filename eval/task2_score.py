#!/usr/bin/env python3
"""
Task 2 evaluation script: tooth registration.

Usage
-----
    python score.py <input_dir> <output_dir>

Expected Codabench-style input layout
-------------------------------------
    <input_dir>/
      ref/
        003/
          upper_gt.npy
          lower_gt.npy
        ...
      res/
        003/
          upper_gt.npy
          lower_gt.npy
        ...

Each .npy file must contain one 4x4 transformation matrix. The script evaluates
upper and lower matrices for every reference case. Extra submitted cases are
ignored; missing submitted cases/files are counted as failed pairs.

Output
------
    <output_dir>/scores.json

The output JSON contains:
    Mean_Translation_Error_mm
    Mean_Rotation_Error_deg
    Num_Evaluated_Pairs / Num_Successful_Pairs / Num_Failed_Pairs
    Overall_Status
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation as R


def is_nan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def calculate_rotation_error_deg(r_pred: np.ndarray, r_gt: np.ndarray) -> float:
    if r_pred.shape != (3, 3) or r_gt.shape != (3, 3):
        return np.nan
    try:
        r_rel = r_pred @ r_gt.T
        return float(np.degrees(R.from_matrix(r_rel).magnitude()))
    except Exception:
        return np.nan


def calculate_translation_error_mm(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    if t_pred.shape != (3,) or t_gt.shape != (3,):
        return np.nan
    try:
        return float(euclidean(t_pred.astype(float), t_gt.astype(float)))
    except Exception:
        return np.nan


def evaluate_pair(pred_path: Path, gt_path: Path) -> dict:
    result = {
        "trans_err_mm": np.nan,
        "rot_err_deg": np.nan,
        "status": "Error",
        "error_message": "",
    }
    try:
        if not pred_path.is_file():
            raise FileNotFoundError(f"Prediction file missing: {pred_path}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"Ground truth file missing: {gt_path}")

        pred = np.load(pred_path)
        gt = np.load(gt_path)
        if pred.shape != (4, 4):
            raise ValueError(f"Prediction matrix shape is {pred.shape}, expected (4, 4)")
        if gt.shape != (4, 4):
            raise ValueError(f"Ground truth matrix shape is {gt.shape}, expected (4, 4)")

        result["trans_err_mm"] = calculate_translation_error_mm(pred[:3, 3], gt[:3, 3])
        result["rot_err_deg"] = calculate_rotation_error_deg(pred[:3, :3], gt[:3, :3])
        if not is_nan(result["trans_err_mm"]) and not is_nan(result["rot_err_deg"]):
            result["status"] = "Success"
        else:
            result["status"] = "Metric Error"
            result["error_message"] = "Metric calculation produced NaN."
    except FileNotFoundError as exc:
        result["status"] = "File Missing"
        result["error_message"] = str(exc)
    except ValueError as exc:
        result["status"] = "Data Error"
        result["error_message"] = str(exc)
    except Exception as exc:
        result["status"] = "System Error"
        result["error_message"] = str(exc)
    return result


def find_case_dirs(root: Path) -> dict:
    if not root.is_dir():
        return {}
    direct = {p.name: p for p in root.iterdir() if p.is_dir()}
    if direct:
        return direct
    labels_dir = root / "labels"
    if labels_dir.is_dir():
        return {p.name: p for p in labels_dir.iterdir() if p.is_dir()}
    return {}


def write_scores(output_dir: Path, scores: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=4)


def run_evaluation(input_dir: Path, output_dir: Path) -> None:
    ref_dir = input_dir / "ref"
    res_dir = input_dir / "res"
    ref_cases = find_case_dirs(ref_dir)
    res_cases = find_case_dirs(res_dir)

    if not ref_cases:
        write_scores(output_dir, {"error": f"No reference case directories found under {ref_dir}"})
        return

    detailed = []
    errors_found = False
    for case_id in sorted(ref_cases):
        for jaw_type in ("upper", "lower"):
            filename = f"{jaw_type}_gt.npy"
            result = evaluate_pair(res_cases.get(case_id, res_dir / case_id) / filename, ref_cases[case_id] / filename)
            result["case_id"] = case_id
            result["jaw_type"] = jaw_type
            detailed.append(result)
            if result["status"] != "Success":
                errors_found = True

    trans_errors = [r["trans_err_mm"] for r in detailed if not is_nan(r["trans_err_mm"])]
    rot_errors = [r["rot_err_deg"] for r in detailed if not is_nan(r["rot_err_deg"])]
    num_total = len(detailed)
    num_success = sum(1 for r in detailed if r["status"] == "Success")
    num_failed = num_total - num_success

    scores = {
        "Mean_Translation_Error_mm": float(np.mean(trans_errors)) if trans_errors else None,
        "Mean_Rotation_Error_deg": float(np.mean(rot_errors)) if rot_errors else None,
        "Num_Evaluated_Pairs": num_total,
        "Num_Total_Pairs": num_total,
        "Num_Successful_Pairs": num_success,
        "Num_Failed_Pairs": num_failed,
        "Overall_Status": "Completed with Errors" if errors_found or num_failed else "Success",
    }
    write_scores(output_dir, scores)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python score.py <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)
    start = time.time()
    run_evaluation(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Evaluation completed in {time.time() - start:.2f} seconds.")
