# src/evaluation/correlation.py
from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


@dataclass
class CorrelationReport:
    cohens_kappa_resolution: float
    cohens_kappa_compliance: float
    spearman_overall_scores: float
    pearson_overall_scores: float
    dimension_correlations: dict[str, float]
    agreement_summary: str


def run_correlation_analysis(
    human_resolution_labels: list[str],
    llm_resolution_labels: list[str],
    human_compliance_flags: list[bool],
    llm_compliance_flags: list[bool],
    human_overall_scores: list[float],
    llm_overall_scores: list[float],
    human_dimension_scores: dict[str, list[float]],
    llm_dimension_scores: dict[str, list[float]],
) -> CorrelationReport:
    kappa_resolution = float(cohen_kappa_score(human_resolution_labels, llm_resolution_labels))
    kappa_compliance = float(cohen_kappa_score(human_compliance_flags, llm_compliance_flags))

    spearman_overall, _ = spearmanr(human_overall_scores, llm_overall_scores)
    pearson_overall, _ = pearsonr(human_overall_scores, llm_overall_scores)

    dim_correlations: dict[str, float] = {}
    for dim in human_dimension_scores:
        rho, _ = spearmanr(human_dimension_scores[dim], llm_dimension_scores[dim])
        dim_correlations[dim] = float(rho)

    lines: list[str] = []
    lines.append(f"Resolution agreement (Cohen's kappa): {kappa_resolution:.3f}")
    lines.append(f"Compliance agreement (Cohen's kappa): {kappa_compliance:.3f}")
    lines.append(f"Overall score Spearman rho: {spearman_overall:.3f}")
    lines.append(f"Overall score Pearson r: {pearson_overall:.3f}")
    for dim, rho in dim_correlations.items():
        lines.append(f"  {dim} Spearman rho: {rho:.3f}")

    return CorrelationReport(
        cohens_kappa_resolution=kappa_resolution,
        cohens_kappa_compliance=kappa_compliance,
        spearman_overall_scores=float(spearman_overall),
        pearson_overall_scores=float(pearson_overall),
        dimension_correlations=dim_correlations,
        agreement_summary="\n".join(lines),
    )
