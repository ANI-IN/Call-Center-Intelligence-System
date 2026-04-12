# src/evaluation/metrics.py
from __future__ import annotations

from dataclasses import dataclass

import jiwer
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error


@dataclass
class TranscriptionMetrics:
    wer: float
    der: float


@dataclass
class SummaryMetrics:
    rouge_1: float
    rouge_2: float
    rouge_l: float
    bertscore_f1: float
    entity_precision: float
    entity_recall: float
    schema_pass_rate: float


@dataclass
class QAMetrics:
    mae_per_dimension: dict[str, float]
    mae_overall: float
    spearman_rho: float
    spearman_p_value: float
    compliance_precision: float
    compliance_recall: float
    compliance_confusion_matrix: list[list[int]]


@dataclass
class CorrelationMetrics:
    cohens_kappa_resolution: float
    cohens_kappa_compliance: float
    spearman_scores: float
    pearson_scores: float


def compute_wer(reference: str, hypothesis: str) -> float:
    return jiwer.wer(reference, hypothesis)


def compute_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge_1": scores["rouge1"].fmeasure,
        "rouge_2": scores["rouge2"].fmeasure,
        "rouge_l": scores["rougeL"].fmeasure,
    }


def compute_bertscore(references: list[str], hypotheses: list[str]) -> float:
    _p, _r, f1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)
    return float(f1.mean())


def compute_entity_metrics(
    reference_entities: list[str], predicted_entities: list[str]
) -> tuple[float, float]:
    ref_set = set(reference_entities)
    pred_set = set(predicted_entities)
    if not pred_set:
        return 0.0, 0.0
    if not ref_set:
        return 0.0, 0.0
    tp = len(ref_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ref_set) if ref_set else 0.0
    return precision, recall


def compute_qa_mae(
    human_scores: dict[str, list[int]], agent_scores: dict[str, list[int]]
) -> dict[str, float]:
    results: dict[str, float] = {}
    for dim in human_scores:
        results[dim] = float(mean_absolute_error(human_scores[dim], agent_scores[dim]))
    return results


def compute_spearman(human: list[float], agent: list[float]) -> tuple[float, float]:
    rho, p_value = spearmanr(human, agent)
    return float(rho), float(p_value)


def compute_compliance_metrics(
    human_flags: list[bool], agent_flags: list[bool]
) -> tuple[float, float, list[list[int]]]:
    cm = confusion_matrix(human_flags, agent_flags, labels=[True, False]).tolist()
    tp = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
    fn = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
    fp = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, cm


def compute_cohens_kappa(human_labels: list, agent_labels: list) -> float:
    return float(cohen_kappa_score(human_labels, agent_labels))
