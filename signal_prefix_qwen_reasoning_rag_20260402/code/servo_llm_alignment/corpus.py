from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .text_templates import build_text_views
from servo_diagnostic.multimodal_method import load_multimodal_arrays


def build_alignment_corpus(
    dataset_path: Path,
    metadata_path: Path | None,
    output_path: Path,
    limit: int | None = None,
) -> Path:
    arrays = load_multimodal_arrays(dataset_path)
    metadata_rows = None
    if metadata_path is not None and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            metadata_rows = list(csv.DictReader(handle))

    scenario_names = arrays["scenario_names"].astype(str)
    family_label_vocab = arrays["family_names"].astype(str)
    location_label_vocab = arrays["location_names"].astype(str)
    boundary_label_vocab = arrays["boundary_names"].astype(str)
    family_names = family_label_vocab[arrays["y_family"].astype(np.int64)]
    location_names = location_label_vocab[arrays["y_loc"].astype(np.int64)]
    boundary_names = boundary_label_vocab[arrays["y_boundary"].astype(np.int64)]
    severity_values = arrays["y_sev"].astype(float)

    count = scenario_names.shape[0] if limit is None else min(limit, int(scenario_names.shape[0]))
    if metadata_rows is not None and len(metadata_rows) != int(scenario_names.shape[0]):
        raise ValueError(
            "Metadata row count does not match dataset size: "
            f"metadata={len(metadata_rows)}, dataset={int(scenario_names.shape[0])}. "
            "Rebuild metadata and dataset together before generating the alignment corpus."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in range(count):
            scenario = str(scenario_names[idx])
            text_views = build_text_views(scenario)
            record = {
                "index": idx,
                "scenario": scenario,
                "family": str(family_names[idx]),
                "location": str(location_names[idx]),
                "boundary": str(boundary_names[idx]),
                "severity": float(severity_values[idx]),
                "condition_name": metadata_rows[idx]["condition_name"] if metadata_rows is not None else "",
                "source_scenario": metadata_rows[idx]["source_scenario"] if metadata_rows is not None else scenario,
                "texts": text_views,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
