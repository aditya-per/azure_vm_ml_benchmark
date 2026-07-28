"""
create_splits.py — Train / Validation / Test split manifests
=============================================================

1. ENTITY split (by vm_id) — used by RQ2 (secondary), RQ3, RQ4 70% train / 10% validation / 20% test, fixed seed.
2. TEMPORAL split (by hour_index) — used by RQ1 (primary), RQ2 (primary) First 600 hours = train, next 120 hours = test (25 days / 5 days),
 with a rolling 3-fold variant for robustness.

Usage
    python create_splits.py
    python create_splits.py --val_split 0.10 --test_split 0.20 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
DEFAULT_INPUT = Path("data/processed/hourly_n2000.parquet")
DEFAULT_OUTDIR = Path("data/processed")
DEFAULT_REPORTDIR = Path("data/reports")

TRACE_HOURS = 720                 # 30 days x 24 hours

# Temporal split (single primary fold)
TRAIN_HOURS = 600                 # first 25 days
TEST_HOURS = 120                  # last 5 days

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("create_splits")


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--reportdir", type=Path, default=DEFAULT_REPORTDIR)
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--test_split", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rolling-folds", type=int, default=3,
                   help="Number of rolling temporal folds for RQ1 robustness")
    return p.parse_args()


# ============================================================
# STEP 1 — Load VM universe
# ============================================================
def load_vm_ids(input_path: Path) -> np.ndarray:
    if not input_path.exists():
        raise FileNotFoundError(f"Input Parquet not found: {input_path}")
    df = pd.read_parquet(input_path, columns=["vm_id"])
    vm_ids = np.sort(df["vm_id"].unique())
    log.info("Loaded %d unique VMs", len(vm_ids))
    return vm_ids


# ============================================================
# STEP 2 — Entity split (by vm_id)
# ============================================================
def make_entity_split(vm_ids: np.ndarray,
                      val_split: float,
                      test_split: float,
                      seed: int) -> pd.DataFrame:
    """
    Deterministic 70/10/20 (default) train/val/test split of VM IDs.
    A VM is assigned to exactly ONE partition (no leakage).
    """
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(vm_ids)

    n = len(shuffled)
    n_test = int(round(test_split * n))
    n_val = int(round(val_split * n))
    n_train = n - n_val - n_test

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]

    manifest = pd.concat([
        pd.DataFrame({"vm_id": np.sort(train_ids), "split": "train"}),
        pd.DataFrame({"vm_id": np.sort(val_ids), "split": "val"}),
        pd.DataFrame({"vm_id": np.sort(test_ids), "split": "test"}),
    ], ignore_index=True)

    manifest["vm_id"] = manifest["vm_id"].astype("int32")
    manifest["split"] = manifest["split"].astype("category")

    log.info("Entity split — train=%d, val=%d, test=%d",
             n_train, n_val, n_test)
    return manifest


# ============================================================
# STEP 3 — Temporal split (by hour_index)
# ============================================================
def make_temporal_split(rolling_folds: int) -> pd.DataFrame:
    """
    Primary fold (fold 0): first 600 hours train, next 120 hours test.

    Rolling folds (1..rolling_folds-1): slide a 20-day train / 5-day test
    window across the 30-day timeline so RQ1 can report mean MAPE across
    multiple test windows covering early / mid / late trace periods.
    """
    rows = []

    # --- Primary fold (fold 0): 25-day train / 5-day test ---
    rows.append({
        "fold": 0, "phase": "train",
        "hour_start": 0, "hour_end": TRAIN_HOURS,          # 0..599
    })
    rows.append({
        "fold": 0, "phase": "test",
        "hour_start": TRAIN_HOURS, "hour_end": TRAIN_HOURS + TEST_HOURS,  # 600..719
    })

    # --- Rolling folds for robustness ---
    # Each fold: 20-day (480h) train, 5-day (120h) test, sliding by ~5 days.
    roll_train = 20 * 24     # 480
    roll_test = 5 * 24       # 120
    stride = 5 * 24          # slide 5 days each fold

    for fold_i in range(1, rolling_folds):
        train_start = (fold_i - 1) * stride
        train_end = train_start + roll_train
        test_start = train_end
        test_end = test_start + roll_test

        if test_end > TRACE_HOURS:
            log.warning("Rolling fold %d exceeds trace length; skipping",
                        fold_i)
            break

        rows.append({
            "fold": fold_i, "phase": "train",
            "hour_start": train_start, "hour_end": train_end,
        })
        rows.append({
            "fold": fold_i, "phase": "test",
            "hour_start": test_start, "hour_end": test_end,
        })

    manifest = pd.DataFrame(rows)
    manifest = manifest.astype({
        "fold": "int8", "hour_start": "int16", "hour_end": "int16",
    })
    manifest["phase"] = manifest["phase"].astype("category")

    log.info("Temporal split — %d folds (fold 0 = primary 25/5-day split)",
             manifest["fold"].nunique())
    return manifest


# ============================================================
# STEP 4 — Sanity report
# ============================================================
#def build_report(entity: pd.DataFrame, temporal: pd.DataFrame, seed: int) -> listlines: list[str] = []
def build_report(entity: pd.DataFrame, temporal: pd.DataFrame, seed: int) -> list[str]:
    lines: list[str] = []
    lines.append("SPLIT MANIFEST REPORT")
    lines.append("=" * 40)
    lines.append(f"Random seed: {seed}")
    lines.append("")

    lines.append("ENTITY SPLIT (by vm_id)")
    counts = entity["split"].value_counts()
    total = len(entity)
    for split_name in ["train", "val", "test"]:
        n = int(counts.get(split_name, 0))
        lines.append(f"  {split_name:6s}: {n:5d} VMs  ({100*n/total:5.1f}%)")
    lines.append(f"  total : {total:5d} VMs")

    # Leakage check
    overlap = entity["vm_id"].duplicated().any()
    lines.append(f"  VM assigned to >1 split? {'YES (ERROR)' if overlap else 'No'}")

    lines.append("")
    lines.append("TEMPORAL SPLIT (by hour_index)")
    for fold in sorted(temporal["fold"].unique()):
        sub = temporal[temporal["fold"] == fold]
        tr = sub[sub["phase"] == "train"].iloc[0]
        te = sub[sub["phase"] == "test"].iloc[0]
        lines.append(
            f"  fold {fold}:  train hours [{tr.hour_start:3d}, {tr.hour_end:3d})"
            f"   test hours [{te.hour_start:3d}, {te.hour_end:3d})"
        )
        # Leakage check: test must start at/after train end
        if te.hour_start < tr.hour_end:
            lines.append(f"    WARNING: fold {fold} temporal leakage detected!")

    return lines


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.reportdir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 55)
    log.info("Creating split manifests")
    log.info("Input: %s", args.input)
    log.info("Seed:  %d", args.seed)
    log.info("=" * 55)

    vm_ids = load_vm_ids(args.input)

    entity = make_entity_split(
        vm_ids, args.val_split, args.test_split, args.seed
    )
    temporal = make_temporal_split(args.rolling_folds)

    entity_path = args.outdir / "split_entity_manifest.parquet"
    temporal_path = args.outdir / "split_temporal_manifest.parquet"
    entity.to_parquet(entity_path, index=False)
    temporal.to_parquet(temporal_path, index=False)
    log.info("Saved entity manifest:   %s", entity_path)
    log.info("Saved temporal manifest: %s", temporal_path)

    lines = build_report(entity, temporal, args.seed)
    report_path = args.reportdir / "split_report.txt"
    report_path.write_text("\n".join(lines))
    log.info("Saved report: %s", report_path)
    for line in lines:
        log.info("  %s", line)

    log.info("Done.")


if __name__ == "__main__":
    main()