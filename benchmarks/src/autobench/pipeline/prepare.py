"""Shared data preparation: task CSV creation, split generation, orchestration.

Framework-specific preparation lives in each adapter:
- ``clam/prepare.py``: H5 -> PT conversion
- ``nnmil/prepare.py``: dataset.json, dataset_plan.json generation
- ``smmile/prepare.py``: H5 -> NIC conversion, superpixel generation
"""

from __future__ import annotations

import os

import pandas as pd

from autobench.config import DatasetConfig
from autobench.data import load_all_slides
from autobench.pipeline.splits import create_strategy_splits


# ---------------------------------------------------------------------------
# Task CSV generation
# ---------------------------------------------------------------------------


def create_task_csv(
    mapping_csv: str,
    output_csv: str,
    label_col: str,
    label_map: dict[int, str],
    ds: DatasetConfig,
) -> pd.DataFrame:
    """Create a CLAM-compatible task CSV from mapping.csv.

    Loads all slides and maps labels using ``label_map``.
    """
    df = load_all_slides(mapping_csv, ds)

    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # Handle label values that may be strings (multiclass) or numeric
    slide_col = ds.slide_id_column
    case_col = ds.case_id_column

    if _is_numeric_labels(df[label_col]):
        raw_labels = df[label_col].astype(int)
        mapped_labels = raw_labels.map(label_map)
        unmapped_mask = mapped_labels.isna()
        if unmapped_mask.any():
            dropped_unmapped = int(unmapped_mask.sum())
            unmapped_counts = raw_labels[unmapped_mask].value_counts(
                dropna=False
            ).to_dict()
            print(
                f"  Dropping {dropped_unmapped} slides with unmapped "
                f"{label_col} values: {unmapped_counts}"
            )
            df = df.loc[~unmapped_mask].copy()
            mapped_labels = mapped_labels.loc[~unmapped_mask]
        task_df = pd.DataFrame({
            "case_id": df[case_col],
            "slide_id": df[slide_col].apply(ds.get_slide_id),
            "label": mapped_labels,
        })
    else:
        # Labels are already strings (e.g., CLWD where CSV has "Acinar", "Solid", etc.)
        # label_map is {0: "Acinar", ...} -- we need reverse: {"Acinar": "Acinar", ...}
        # Just use the raw label values directly since they are already class names
        allowed_labels = set(label_map.values())
        known_mask = df[label_col].isin(allowed_labels)
        if (~known_mask).any():
            dropped_unmapped = int((~known_mask).sum())
            unmapped_counts = df.loc[~known_mask, label_col].value_counts(
                dropna=False
            ).to_dict()
            print(
                f"  Dropping {dropped_unmapped} slides with unmapped "
                f"{label_col} values: {unmapped_counts}"
            )
            df = df.loc[known_mask].copy()
        task_df = pd.DataFrame({
            "case_id": df[case_col],
            "slide_id": df[slide_col].apply(ds.get_slide_id),
            "label": df[label_col],
        })

    if task_df.empty:
        raise ValueError(
            f"Task {label_col!r} has no labelled slides after applying "
            f"label_map={label_map}. Check the dataset YAML and mapping CSV."
        )
    label_counts = task_df["label"].value_counts(dropna=False).to_dict()
    if task_df["label"].isna().any():
        raise ValueError(
            f"Task {label_col!r} still contains missing labels after "
            f"filtering. Label counts: {label_counts}"
        )
    if len(label_counts) < 2:
        raise ValueError(
            f"Task {label_col!r} has only one class after filtering: "
            f"{label_counts}. Check the dataset YAML and mapping CSV."
        )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    task_df.to_csv(output_csv, index=False)
    print(f"  Task CSV: {output_csv}  ({len(task_df)} slides, "
          f"{label_counts})")
    return task_df


def _is_numeric_labels(series: pd.Series) -> bool:
    """Check if a label column contains numeric values."""
    try:
        series.dropna().astype(float)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Full preparation orchestrator
# ---------------------------------------------------------------------------


def prepare_all(
    benchmark_dir: str,
    mapping_csv: str,
    features_base_dir: str,
    encoder_keys: list[str],
    ds: DatasetConfig,
    seed: int = 42,
    n_splits: int = 5,
    task_names: list[str] | None = None,
) -> None:
    """Run the complete data-preparation pipeline (idempotent).

    Uses task definitions from ``DatasetConfig`` instead of hardcoded values.
    """
    all_slide_ids: set[str] = set()

    # Get default strategy (first one defined in the config)
    default_strategy = list(ds.split_strategies.keys())[0]

    task_names = task_names or list(ds.tasks.keys())
    unknown_tasks = [
        task_name for task_name in task_names if task_name not in ds.tasks
    ]
    if unknown_tasks:
        raise ValueError(
            f"Unknown task(s) requested for preparation: {unknown_tasks}. "
            f"Valid tasks: {list(ds.tasks.keys())}"
        )

    for task_name in task_names:
        tdef = ds.tasks[task_name]
        csv_path = os.path.join(benchmark_dir, "dataset_csv", f"{task_name}.csv")
        if not os.path.exists(csv_path):
            print(f"[prep] Creating task CSV: {task_name}")
            task_df = create_task_csv(
                mapping_csv, csv_path,
                label_col=tdef.label_col,
                label_map=tdef.label_map,
                ds=ds,
            )
        else:
            print(f"[prep] Task CSV already exists: {csv_path}")
            task_df = pd.read_csv(csv_path)
            if task_df["label"].isna().any():
                print(
                    f"[prep] Existing task CSV has missing labels; "
                    f"regenerating: {task_name}"
                )
                task_df = create_task_csv(
                    mapping_csv, csv_path,
                    label_col=tdef.label_col,
                    label_map=tdef.label_map,
                    ds=ds,
                )
        all_slide_ids.update(task_df["slide_id"].tolist())

        splits_dir = os.path.join(benchmark_dir, "splits", default_strategy, task_name)
        first_split = os.path.join(splits_dir, "splits_0.csv")
        if not os.path.exists(first_split):
            print(f"[prep] Creating splits: {task_name}")
            create_strategy_splits(csv_path, splits_dir, n_splits=n_splits, seed=seed)
        else:
            print(f"[prep] Splits already exist: {splits_dir}")

    # H5 -> PT for each encoder (CLAM-specific)
    from autobench.pipeline.clam.prepare import convert_h5_to_pt

    slide_ids_sorted = sorted(all_slide_ids)
    for encoder_key in encoder_keys:
        h5_dir = os.path.join(features_base_dir, f"features_{encoder_key}")
        pt_dir = os.path.join(benchmark_dir, "features", encoder_key)
        print(f"[prep] Converting H5->PT: {encoder_key} ({len(slide_ids_sorted)} slides)")
        n = convert_h5_to_pt(h5_dir, pt_dir, encoder_key, slide_ids_sorted)
        if n > 0:
            print(f"  Converted {n} new files")
        else:
            print(f"  All files already converted")
