from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from modality_evidence_topk_v2 import (
    MODALITY_ORDER,
    attention_entropy,
    get_prior_for_scenario,
    normalize_modality_name,
    normalize_scenario_name,
    topk_contains,
    weighted_consistency_at_3,
)

VIEWS = ["evidence", "mechanism"]


def _find_scenario(row: dict[str, str]) -> str:
    candidates = [
        "true_class_name",
        "target_result",
        "scenario",
        "target_class_name",
        "true_label_name",
        "label_name",
    ]
    for key in candidates:
        value = row.get(key, "").strip()
        if value:
            return value
    return "unknown"


def _find_sample_index(row: dict[str, str], default_index: int) -> str:
    for key in ["sample_index", "index", "row_index"]:
        value = row.get(key, "").strip()
        if value:
            return value
    return str(default_index)


def _candidate_keys(view: str, modality: str) -> list[str]:
    return [
        f"{view}_{modality}_score",
        f"{view}_score_{modality}",
        f"score_{view}_{modality}",
        f"{modality}_{view}_score",
        f"{view}_{modality}",
        f"{modality}_{view}",
    ]


def _extract_scores(row: dict[str, str], view: str) -> dict[str, float] | None:
    lowered = {k.lower(): v for k, v in row.items()}
    found: dict[str, float] = {}
    for modality in MODALITY_ORDER:
        value = None
        for key in _candidate_keys(view, modality):
            if key in lowered and str(lowered[key]).strip() != "":
                value = lowered[key]
                break
        if value is None:
            return None
        found[modality] = float(value)
    return found


def _rank_modalities(scores: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def analyze_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_rows: list[dict[str, Any]] = []
    per_view_stats: dict[str, dict[str, list[float]]] = {view: defaultdict(list) for view in VIEWS}
    per_class_view_stats: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: {view: defaultdict(list) for view in VIEWS})
    normal_entropy: dict[str, list[float]] = defaultdict(list)

    for idx, row in enumerate(rows):
        scenario = _find_scenario(row)
        scenario_key = normalize_scenario_name(scenario)
        prior = get_prior_for_scenario(scenario)
        record: dict[str, Any] = {
            "sample_index": _find_sample_index(row, idx),
            "scenario": scenario,
            "scenario_key": scenario_key,
            "primary_modalities": list(prior.primary),
            "support_modalities": list(prior.support),
        }

        for view in VIEWS:
            scores = _extract_scores(row, view)
            if scores is None:
                continue
            rank_modalities = _rank_modalities(scores)
            probs = [scores[m] for m in MODALITY_ORDER]
            record[f"{view}_top1"] = rank_modalities[0]
            record[f"{view}_top2"] = ",".join(rank_modalities[:2])
            record[f"{view}_top3"] = ",".join(rank_modalities[:3])
            record[f"{view}_primary_in_top2"] = int(topk_contains(rank_modalities[:2], prior.primary))
            record[f"{view}_primary_in_top3"] = int(topk_contains(rank_modalities[:3], prior.primary))
            record[f"{view}_primary_or_support_hit_at_2"] = int(topk_contains(rank_modalities[:2], list(prior.primary) + list(prior.support)))
            record[f"{view}_primary_or_support_hit_at_3"] = int(topk_contains(rank_modalities[:3], list(prior.primary) + list(prior.support)))
            record[f"{view}_weighted_consistency_at_3"] = weighted_consistency_at_3(rank_modalities[:3], prior)
            record[f"{view}_attention_entropy"] = attention_entropy(probs)

            per_view_stats[view]["primary_in_top2"].append(float(record[f"{view}_primary_in_top2"]))
            per_view_stats[view]["primary_in_top3"].append(float(record[f"{view}_primary_in_top3"]))
            per_view_stats[view]["primary_or_support_hit_at_2"].append(float(record[f"{view}_primary_or_support_hit_at_2"]))
            per_view_stats[view]["primary_or_support_hit_at_3"].append(float(record[f"{view}_primary_or_support_hit_at_3"]))
            per_view_stats[view]["weighted_consistency_at_3"].append(float(record[f"{view}_weighted_consistency_at_3"]))
            if scenario_key == "normal":
                normal_entropy[view].append(float(record[f"{view}_attention_entropy"]))
            else:
                per_class_view_stats[scenario_key][view]["primary_in_top2"].append(float(record[f"{view}_primary_in_top2"]))
                per_class_view_stats[scenario_key][view]["primary_or_support_hit_at_2"].append(float(record[f"{view}_primary_or_support_hit_at_2"]))
                per_class_view_stats[scenario_key][view]["weighted_consistency_at_3"].append(float(record[f"{view}_weighted_consistency_at_3"]))

        sample_rows.append(record)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    summary: dict[str, Any] = {
        "views": {},
        "per_class_fault_only": {},
        "normal": {},
        "modality_order": list(MODALITY_ORDER),
        "notes": {
            "fault_only": "Per-class metrics exclude normal from fault evidence consistency.",
            "normal": "Normal is tracked with attention entropy instead of top-k hit logic.",
        },
    }

    for view in VIEWS:
        stats = per_view_stats[view]
        summary["views"][view] = {
            "primary_in_top2": _mean(stats["primary_in_top2"]),
            "primary_in_top3": _mean(stats["primary_in_top3"]),
            "primary_or_support_hit_at_2": _mean(stats["primary_or_support_hit_at_2"]),
            "primary_or_support_hit_at_3": _mean(stats["primary_or_support_hit_at_3"]),
            "weighted_consistency_at_3": _mean(stats["weighted_consistency_at_3"]),
        }
        summary["normal"][view] = {
            "attention_entropy": _mean(normal_entropy[view]),
            "sample_count": len(normal_entropy[view]),
        }

    for scenario_key, scenario_stats in per_class_view_stats.items():
        summary["per_class_fault_only"][scenario_key] = {}
        for view in VIEWS:
            metrics = scenario_stats[view]
            summary["per_class_fault_only"][scenario_key][view] = {
                "primary_in_top2": _mean(metrics["primary_in_top2"]),
                "primary_or_support_hit_at_2": _mean(metrics["primary_or_support_hit_at_2"]),
                "weighted_consistency_at_3": _mean(metrics["weighted_consistency_at_3"]),
                "sample_count": len(metrics["primary_in_top2"]),
            }

    return sample_rows, summary


def write_outputs(output_dir: Path, sample_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "test_modality_evidence_topk_scores_v2.csv"
    json_path = output_dir / "test_modality_evidence_topk_summary_v2.json"

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in sample_rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"topk_csv={csv_path}")
    print(f"topk_summary={json_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.scores_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    sample_rows, summary = analyze_rows(rows)
    write_outputs(args.output_dir, sample_rows, summary)


if __name__ == "__main__":
    main()
