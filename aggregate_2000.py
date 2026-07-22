#!/usr/bin/env python3
"""
Usage
-----
python aggregate.py --input .\data\raw --output .\data\processed
"""

from pathlib import Path
import argparse
import logging
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SCHEMA = ["encrypted_vmid", "seconds", "avg_cpu"]

DTYPES = {
    "encrypted_vmid": "string",
    "seconds": "int64",
    "avg_cpu": "float32",
}

SECONDS_PER_HOUR = 3600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("aggregate")


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        type=Path,
        default=Path("csv"),
        help="Directory containing CSV files"
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output directory"
    )

    return p.parse_args()


# ---------------------------------------------------------
# Find CSV files
# ---------------------------------------------------------

def find_files(folder: Path):

    files = sorted(folder.glob("*.csv"))
    files += sorted(folder.glob("*.csv.gz"))

    if len(files) == 0:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    log.info(f"Found {len(files)} CSV files")

    return files


# ---------------------------------------------------------
# Pass 1 - discover all VM IDs
# ---------------------------------------------------------

def discover_vm_ids(files):

    vm_ids = set()

    for file in files:

        log.info(f"Scanning {file.name}")

        df = pd.read_csv(
            file,
            header=None,
            names=SCHEMA,
            usecols=[0],
            dtype={"encrypted_vmid": "string"},
            compression="infer"
        )

        vm_ids.update(df["encrypted_vmid"].dropna().unique())

    vm_ids = sorted(vm_ids)

    mapping = pd.DataFrame({
        "vm_id": np.arange(len(vm_ids), dtype=np.int32),
        "encrypted_vmid": vm_ids
    })

    log.info(f"Discovered {len(mapping):,} unique VMs")

    return mapping


# ---------------------------------------------------------
# Pass 2 - Aggregate
# ---------------------------------------------------------

def aggregate(files, mapping):

    lookup = dict(
        zip(
            mapping["encrypted_vmid"],
            mapping["vm_id"]
        )
    )

    keep = set(lookup.keys())

    result = []

    total_rows = 0

    for i, file in enumerate(files, 1):

        log.info(f"[{i}/{len(files)}] {file.name}")

        df = pd.read_csv(
            file,
            header=None,
            names=SCHEMA,
            dtype=DTYPES,
            compression="infer"
        )

        total_rows += len(df)

        df = df[df["encrypted_vmid"].isin(keep)].copy()

        df["vm_id"] = df["encrypted_vmid"].map(lookup)

        # 0-3599 -> hour 0
        # 3600-7199 -> hour 1
        df["hour_index"] = (
            df["seconds"] // SECONDS_PER_HOUR
        ).astype(np.int16)

        hourly = (
            df
            .groupby(
                ["vm_id", "hour_index"],
                sort=True
            )["avg_cpu"]
            .agg(
                avg_cpu_mean="mean",
                avg_cpu_min="min",
                avg_cpu_max="max",
                avg_cpu_std="std",
                readings_count="count"
            )
            .reset_index()
        )

        hourly["avg_cpu_std"] = (
            hourly["avg_cpu_std"]
            .fillna(0)
            .astype(np.float32)
        )

        hourly["hour_of_day"] = (
            hourly["hour_index"] % 24
        ).astype(np.int8)

        hourly["day_index"] = (
            hourly["hour_index"] // 24
        ).astype(np.int8)

        hourly["day_of_cycle"] = (
            hourly["day_index"] % 7
        ).astype(np.int8)

        result.append(hourly)

    hourly = pd.concat(result, ignore_index=True)

    hourly.sort_values(
        ["vm_id", "hour_index"],
        inplace=True
    )

    hourly.reset_index(drop=True, inplace=True)

    log.info(f"Processed {total_rows:,} rows")

    return hourly


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

def write_report(df, outfile):

    hours = df.groupby("vm_id").size()

    lines = []

    lines.append(f"VMs               : {df.vm_id.nunique():,}")
    lines.append(f"Rows              : {len(df):,}")
    lines.append(f"Min hours / VM    : {hours.min()}")
    lines.append(f"Median hours / VM : {int(hours.median())}")
    lines.append(f"Max hours / VM    : {hours.max()}")

    outfile.write_text("\n".join(lines))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    args.output.mkdir(exist_ok=True)

    files = find_files(args.input)

    mapping = discover_vm_ids(files)

    mapping.to_parquet(
        args.output / "vm_id_mapping.parquet",
        index=False
    )

    hourly = aggregate(files, mapping)
    
    # Save Parquet
    parquet_file = (
        args.output /
        f"hourly_n{hourly.vm_id.nunique()}.parquet"
    )

    hourly.to_parquet(
        parquet_file,
        index=False,
        compression="snappy"
    )

    # Save CSV
    csv_file = (
        args.output /
        f"hourly_n{hourly.vm_id.nunique()}.csv"
    )

    hourly.to_csv(
        csv_file,
        index=False
    )

    log.info(f"Saved Parquet: {parquet_file}")
    log.info(f"Saved CSV: {csv_file}")

if __name__ == "__main__":
    main()