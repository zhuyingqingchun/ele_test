from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from servo_diagnostic.feature_engineering import (
    DEFAULT_SIGNAL_COLUMNS,
    WindowConfig,
    build_window_samples,
    save_window_dataset,
    save_window_metadata,
)
from servo_diagnostic.io_utils import save_dataset
from servo_diagnostic.multimodal_method import (
    build_multimodal_window_dataset,
    load_csv_rows,
)


DEFAULT_EXCLUDED_SCENARIOS = ("load_disturbance_mild",)


def parse_name_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def filter_rows(rows: list[dict[str, str]], excluded_scenarios: set[str]) -> list[dict[str, str]]:
    return [row for row in rows if str(row.get("scenario", "")) not in excluded_scenarios]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the recommended handoff dataset bundle for external fault-diagnosis use."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "diagnostic_datasets" / "servo_fault_diagnosis_dataset_full.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "diagnostic_datasets" / "servo_fault_diagnosis_dataset_handoff.csv",
    )
    parser.add_argument(
        "--multimodal-output",
        type=Path,
        default=PROJECT_ROOT / "derived_datasets" / "servo_multimodal_handoff_dataset.npz",
    )
    parser.add_argument(
        "--multimodal-metadata",
        type=Path,
        default=PROJECT_ROOT / "derived_datasets" / "servo_multimodal_handoff_metadata.csv",
    )
    parser.add_argument(
        "--window-output",
        type=Path,
        default=PROJECT_ROOT / "derived_datasets" / "servo_window_handoff_dataset.npz",
    )
    parser.add_argument(
        "--window-metadata",
        type=Path,
        default=PROJECT_ROOT / "derived_datasets" / "servo_window_handoff_metadata.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "derived_datasets" / "servo_handoff_manifest.json",
    )
    parser.add_argument("--exclude-scenarios", type=str, default=",".join(DEFAULT_EXCLUDED_SCENARIOS))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--active-ratio-threshold", type=float, default=0.15)
    parser.add_argument("--keep-ambiguous-windows", action="store_true")
    args = parser.parse_args()

    excluded_scenarios = set(parse_name_list(args.exclude_scenarios))
    rows = load_csv_rows(args.input)
    filtered_rows = filter_rows(rows, excluded_scenarios)
    if not filtered_rows:
        raise ValueError("All rows were filtered out. Adjust --exclude-scenarios.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.multimodal_output.parent.mkdir(parents=True, exist_ok=True)
    args.window_output.parent.mkdir(parents=True, exist_ok=True)

    save_dataset(args.output_csv, filtered_rows)

    multimodal_artifacts = build_multimodal_window_dataset(
        rows=filtered_rows,
        output_path=args.multimodal_output,
        metadata_path=args.multimodal_metadata,
        config=[WindowConfig(window_size=128, stride=32), WindowConfig(window_size=256, stride=64), WindowConfig(window_size=512, stride=128)],
        active_ratio_threshold=args.active_ratio_threshold,
        drop_ambiguous=not args.keep_ambiguous_windows,
    )

    windows, metadata = build_window_samples(
        filtered_rows,
        DEFAULT_SIGNAL_COLUMNS,
        WindowConfig(window_size=args.window_size, stride=args.stride),
        active_ratio_threshold=args.active_ratio_threshold,
        drop_ambiguous=not args.keep_ambiguous_windows,
    )
    save_window_dataset(args.window_output, windows, metadata, DEFAULT_SIGNAL_COLUMNS)
    save_window_metadata(args.window_metadata, metadata)

    handoff_scenarios = sorted({str(item["scenario"]) for item in metadata})
    payload = {
        "input_csv": str(args.input),
        "handoff_csv": str(args.output_csv),
        "multimodal_dataset": str(args.multimodal_output),
        "multimodal_metadata": str(args.multimodal_metadata),
        "window_dataset": str(args.window_output),
        "window_metadata": str(args.window_metadata),
        "excluded_scenarios": sorted(excluded_scenarios),
        "rows_total": len(rows),
        "rows_handoff": len(filtered_rows),
        "window_samples": int(windows.shape[0]),
        "multimodal_windows": int(multimodal_artifacts.num_windows),
        "scenario_classes": handoff_scenarios,
        "window_size": args.window_size,
        "stride": args.stride,
        "active_ratio_threshold": args.active_ratio_threshold,
        "drop_ambiguous_windows": not args.keep_ambiguous_windows,
    }
    args.manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"excluded_scenarios={sorted(excluded_scenarios)}")
    print(f"handoff_csv={args.output_csv}")
    print(f"multimodal_dataset={args.multimodal_output}")
    print(f"window_dataset={args.window_output}")
    print(f"manifest={args.manifest}")
    print(f"rows_handoff={len(filtered_rows)}")
    print(f"multimodal_windows={multimodal_artifacts.num_windows}")
    print(f"window_samples={windows.shape[0]}")


if __name__ == "__main__":
    main()
