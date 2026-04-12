# src/evaluation/run_eval.py
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.evaluation.metrics import (
    compute_bertscore,
    compute_compliance_metrics,
    compute_qa_mae,
    compute_rouge,
    compute_spearman,
    compute_wer,
)


def load_ground_truth(gt_dir: Path) -> list[dict]:
    entries: list[dict] = []
    for f in sorted(gt_dir.glob("*.json")):
        with open(f) as fh:
            entries.append(json.load(fh))
    return entries


def load_thresholds(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["thresholds"]


def run_transcription_eval(ground_truth: list[dict], predictions: list[dict]) -> dict:
    wer_scores: list[float] = []
    for gt, pred in zip(ground_truth, predictions):
        w = compute_wer(gt["reference_transcript"], pred["transcript"])
        wer_scores.append(w)
    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0.0
    return {"avg_wer": round(avg_wer, 4), "per_call_wer": wer_scores}


def run_summary_eval(ground_truth: list[dict], predictions: list[dict]) -> dict:
    rouge_scores: list[dict] = []
    references: list[str] = []
    hypotheses: list[str] = []

    for gt, pred in zip(ground_truth, predictions):
        rouge = compute_rouge(gt["reference_summary"], pred["summary"])
        rouge_scores.append(rouge)
        references.append(gt["reference_summary"])
        hypotheses.append(pred["summary"])

    avg_rouge_l = sum(r["rouge_l"] for r in rouge_scores) / len(rouge_scores)
    bert_f1 = compute_bertscore(references, hypotheses)

    return {
        "avg_rouge_l": round(avg_rouge_l, 4),
        "bertscore_f1": round(bert_f1, 4),
    }


def run_qa_eval(ground_truth: list[dict], predictions: list[dict]) -> dict:
    dimensions = [
        "professionalism",
        "empathy",
        "problem_resolution",
        "compliance",
        "communication_clarity",
    ]
    human_scores: dict[str, list[int]] = {d: [] for d in dimensions}
    agent_scores: dict[str, list[int]] = {d: [] for d in dimensions}
    human_overall: list[float] = []
    agent_overall: list[float] = []
    human_compliance: list[bool] = []
    agent_compliance: list[bool] = []

    for gt, pred in zip(ground_truth, predictions):
        for d in dimensions:
            human_scores[d].append(gt["qa_scores"][d])
            agent_scores[d].append(pred["qa_scores"][d])
        human_overall.append(gt["qa_scores"]["overall"])
        agent_overall.append(pred["qa_scores"]["overall"])
        human_compliance.append(gt.get("has_compliance_violation", False))
        agent_compliance.append(pred.get("has_compliance_violation", False))

    mae = compute_qa_mae(human_scores, agent_scores)
    rho, p = compute_spearman(human_overall, agent_overall)
    c_prec, c_rec, c_cm = compute_compliance_metrics(human_compliance, agent_compliance)

    return {
        "mae_per_dimension": mae,
        "mae_overall": round(sum(mae.values()) / len(mae), 4),
        "spearman_rho": round(rho, 4),
        "compliance_precision": round(c_prec, 4),
        "compliance_recall": round(c_rec, 4),
    }


def check_thresholds(results: dict, thresholds: dict) -> list[str]:
    failures: list[str] = []
    if "avg_wer" in results and results["avg_wer"] > thresholds.get("transcription", {}).get(
        "wer", 1.0
    ):
        failures.append(
            f"WER {results['avg_wer']} exceeds threshold {thresholds['transcription']['wer']}"
        )
    if "avg_rouge_l" in results and results["avg_rouge_l"] < thresholds.get("summary", {}).get(
        "rouge_l", 0.0
    ):
        failures.append(
            f"ROUGE-L {results['avg_rouge_l']} below threshold {thresholds['summary']['rouge_l']}"
        )
    if "bertscore_f1" in results and results["bertscore_f1"] < thresholds.get("summary", {}).get(
        "bertscore_f1", 0.0
    ):
        thresh = thresholds["summary"]["bertscore_f1"]
        failures.append(f"BERTScore F1 {results['bertscore_f1']} below threshold {thresh}")
    qa_thresh = thresholds.get("qa_scoring", {})
    if "spearman_rho" in results and results["spearman_rho"] < qa_thresh.get("spearman_rho", 0.0):
        failures.append(
            f"Spearman rho {results['spearman_rho']} below threshold {qa_thresh['spearman_rho']}"
        )
    if "compliance_recall" in results and results["compliance_recall"] < qa_thresh.get(
        "compliance_recall", 0.0
    ):
        failures.append(
            f"Compliance recall "
            f"{results['compliance_recall']} "
            f"below threshold "
            f"{qa_thresh['compliance_recall']}"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--transcription", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--correlation", action="store_true")
    parser.add_argument("--gt-dir", default="evaluations/ground_truth")
    parser.add_argument("--pred-dir", default="evaluations/results/predictions")
    parser.add_argument("--config", default="evaluations/config.yaml")
    parser.add_argument("--output", default="evaluations/results/eval_results.json")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    config_path = Path(args.config)
    output_path = Path(args.output)

    if not gt_dir.exists():
        print(f"Ground truth directory not found: {gt_dir}")
        sys.exit(1)

    ground_truth = load_ground_truth(gt_dir)
    thresholds = load_thresholds(config_path)

    pred_dir = Path(args.pred_dir)
    predictions: list[dict] = []
    if pred_dir.exists():
        for f in sorted(pred_dir.glob("*.json")):
            with open(f) as fh:
                predictions.append(json.load(fh))

    results: dict = {"timestamp": datetime.now().isoformat(), "metrics": {}}

    if args.all or args.transcription:
        results["metrics"]["transcription"] = run_transcription_eval(ground_truth, predictions)

    if args.all or args.summary:
        results["metrics"]["summary"] = run_summary_eval(ground_truth, predictions)

    if args.all or args.qa:
        results["metrics"]["qa"] = run_qa_eval(ground_truth, predictions)

    all_metrics = {}
    for section in results["metrics"].values():
        all_metrics.update(section)
    failures = check_thresholds(all_metrics, thresholds)
    results["threshold_failures"] = failures
    results["passed"] = len(failures) == 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    if failures:
        print(f"\nFAILED: {len(failures)} threshold(s) not met:")
        for fail in failures:
            print(f"  - {fail}")
        sys.exit(1)
    else:
        print("\nPASSED: All thresholds met.")


if __name__ == "__main__":
    main()
