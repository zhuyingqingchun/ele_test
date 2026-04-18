from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from servo_diagnostic.feature_engineering import WindowConfig
from servo_diagnostic.multimodal_method import build_multimodal_window_dataset, load_csv_rows


def parse_window_configs(window_sizes_arg: str, stride_ratio: float, stride_arg: int | None) -> list[WindowConfig]:
    window_sizes = [int(item.strip()) for item in window_sizes_arg.split(",") if item.strip()]
    if not window_sizes:
        raise ValueError("window_sizes_arg must contain at least one integer window size.")
    configs: list[WindowConfig] = []
    for window_size in window_sizes:
        stride = stride_arg if stride_arg is not None else max(1, int(round(window_size * stride_ratio)))
        configs.append(WindowConfig(window_size=window_size, stride=stride))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the multimodal servo dataset used by the physics-guided reasoning model.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "diagnostic_datasets" / "servo_fault_diagnosis_dataset_full.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "derived_datasets" / "servo_multimodal_dataset.npz")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "derived_datasets" / "servo_multimodal_metadata.csv")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--window-sizes", type=str, default="128,256,512")
    parser.add_argument("--stride-ratio", type=float, default=0.25)
    parser.add_argument("--single-scale", action="store_true")
    parser.add_argument("--active-ratio-threshold", type=float, default=0.15)
    parser.add_argument("--keep-ambiguous-windows", action="store_true")
    args = parser.parse_args()

    rows = load_csv_rows(args.input)
    if args.single_scale:
        configs: WindowConfig | list[WindowConfig] = WindowConfig(
            window_size=args.window_size,
            stride=args.stride if args.stride is not None else max(1, args.window_size // 4),
        )
    else:
        configs = parse_window_configs(args.window_sizes, args.stride_ratio, args.stride)
    artifacts = build_multimodal_window_dataset(
        rows=rows,
        output_path=args.output,
        metadata_path=args.metadata,
        config=configs,
        active_ratio_threshold=args.active_ratio_threshold,
        drop_ambiguous=not args.keep_ambiguous_windows,
    )
    print(f"Windows: {artifacts.num_windows}")
    print(f"Primary window size: {artifacts.window_size}")
    print(f"Primary stride: {artifacts.stride}")
    print(f"Window sizes: {artifacts.window_sizes}")
    print(f"Strides: {artifacts.strides}")
    print(f"Dataset: {artifacts.output_path}")
    print(f"Metadata: {artifacts.metadata_path}")
    print(f"Scenario classes: {len(artifacts.class_names)}")
    print(f"Family classes: {len(artifacts.family_names)}")
    print(f"Location classes: {len(artifacts.location_names)}")
    print(f"Fault-active ratio threshold: {args.active_ratio_threshold}")
    print(f"Drop ambiguous windows: {not args.keep_ambiguous_windows}")


if __name__ == "__main__":
    main()
