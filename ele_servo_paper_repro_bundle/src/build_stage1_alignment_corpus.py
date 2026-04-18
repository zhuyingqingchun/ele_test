from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from servo_diagnostic.multimodal_method import (
    BOUNDARY_BY_SCENARIO,
    FAMILY_BY_SCENARIO,
    LOCATION_BY_SCENARIO,
)
from servo_llm_alignment.template_config import DEFAULT_TEMPLATE_NAME, TEMPLATE_CONFIGS
from servo_llm_alignment.text_templates import available_template_names, build_text_views


def _read_metadata_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _first_present(row: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    for key in candidates:
        if key in row and row[key] not in ("", None):
            return row[key]
    return default


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_alignment_corpus_from_metadata(
    metadata_path: Path,
    output_path: Path,
    *,
    template_name: str = DEFAULT_TEMPLATE_NAME,
    limit: int | None = None,
    seed_offset: int = 0,
    keep_legacy_fields: bool = True,
    minimal_output: bool = False,
) -> Path:
    if template_name not in TEMPLATE_CONFIGS:
        raise ValueError(
            f"Unknown template_name={template_name!r}. Available: {', '.join(available_template_names())}"
        )

    rows = _read_metadata_rows(metadata_path)
    if limit is not None and limit > 0:
        rows = rows[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for local_idx, row in enumerate(rows):
            index = _parse_int(
                _first_present(row, ["index", "sample_index", "window_index", "idx"], default=local_idx),
                default=local_idx,
            )

            scenario = str(
                _first_present(
                    row,
                    ["scenario", "scenario_name", "window_label", "fault_label"],
                    default="normal",
                )
            )

            condition_name = str(
                _first_present(
                    row,
                    ["condition_name", "condition", "operating_condition"],
                    default="",
                )
            )

            source_scenario = str(
                _first_present(
                    row,
                    ["source_scenario", "source_fault_label", "source_label"],
                    default=scenario,
                )
            )

            severity = _parse_float(
                _first_present(row, ["severity", "fault_severity", "y_sev"], default=0.0),
                default=0.0,
            )

            family = str(_first_present(row, ["family"], default=FAMILY_BY_SCENARIO.get(scenario, "unknown")))
            location = str(_first_present(row, ["location"], default=LOCATION_BY_SCENARIO.get(scenario, "unknown")))
            boundary = str(_first_present(row, ["boundary"], default=BOUNDARY_BY_SCENARIO.get(scenario, "unknown")))

            texts = build_text_views(
                scenario,
                condition_name=condition_name or None,
                severity=severity,
                source_scenario=source_scenario or None,
                template_name=template_name,
                seed=seed_offset + index,
            )

            if minimal_output:
                record = {
                    "index": index,
                    "scenario": scenario,
                    "text": texts["combined_text"],
                    "template_name": texts["template_name"],
                }
            else:
                record = {
                    "index": index,
                    "scenario": scenario,
                    "family": family,
                    "location": location,
                    "boundary": boundary,
                    "severity": severity,
                    "condition_name": condition_name,
                    "source_scenario": source_scenario,
                    "text": texts["combined_text"],
                    "template_name": texts["template_name"],
                    "pieces": [p.strip() for p in texts["combined_text"].replace(".", ";").split(";") if p.strip()],
                }

                if keep_legacy_fields:
                    record["texts"] = {
                        "scenario_text": texts["scenario_text"],
                        "family_text": texts["family_text"],
                        "location_text": texts["location_text"],
                        "boundary_text": texts["boundary_text"],
                        "mechanism_text": texts["mechanism_text"],
                        "evidence_text": texts["evidence_text"],
                        "contrast_text": texts["contrast_text"],
                        "combined_text": texts["combined_text"],
                    }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stage-1 signal-text alignment corpus from metadata with configurable text strategies."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "derived_datasets" / "servo_multimodal_handoff_dataset.npz",
        help="Reserved for compatibility; not required by this standalone builder.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "derived_datasets" / "servo_multimodal_handoff_metadata.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "derived_datasets" / "stage1_alignment_corpus.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--template-name",
        type=str,
        default=DEFAULT_TEMPLATE_NAME,
        choices=available_template_names(),
        help="Text generation strategy name.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Added to sample index to control per-sample phrase sampling.",
    )
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Only keep index/scenario/text/template_name in output JSONL.",
    )
    parser.add_argument(
        "--no-legacy-fields",
        action="store_true",
        help="Do not store the nested texts dict for backward compatibility.",
    )
    parser.add_argument(
        "--print-template-options",
        action="store_true",
        help="Print available template names and exit.",
    )
    args = parser.parse_args()

    if args.print_template_options:
        print(json.dumps({"available_template_names": available_template_names()}, ensure_ascii=False, indent=2))
        return

    limit = None if args.limit <= 0 else args.limit

    output = build_alignment_corpus_from_metadata(
        metadata_path=args.metadata,
        output_path=args.output,
        template_name=args.template_name,
        limit=limit,
        seed_offset=args.seed_offset,
        keep_legacy_fields=not args.no_legacy_fields,
        minimal_output=args.minimal_output,
    )
    print(f"corpus={output}")
    print(f"template_name={args.template_name}")


if __name__ == "__main__":
    main()