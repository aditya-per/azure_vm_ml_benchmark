#!/usr/bin/env python3

from pathlib import Path
from zipfile import ZipFile
import argparse
import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SCHEMA = [
    "encrypted_vmid",
    "seconds",
    "avg_cpu"
]

DTYPES = {
    "encrypted_vmid": "string",
    "seconds": "int64",
    "avg_cpu": "float32"
}

SECONDS_PER_HOUR = 3600

EXPECTED_POINTS = 8640
TARGET_VM_COUNT = 2000
RANDOM_SEED = 42

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
        default=Path("data/raw"),
        help="Directory containing raw files"
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory"
    )

    return p.parse_args()


# ---------------------------------------------------------
# File Discovery
# ---------------------------------------------------------

def find_files(folder: Path):

    files = sorted(folder.glob("*.csv"))
    files += sorted(folder.glob("*.csv.gz"))
    files += sorted(folder.glob("*.zip"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}"
        )

    log.info(f"Found {len(files)} files")

    return files


# ---------------------------------------------------------
# Read CSV / ZIP
# ---------------------------------------------------------

def read_csv_file(file, **kwargs):

    if file.suffix.lower() == ".zip":

        with ZipFile(file) as z:

            csv_members = [
                m
                for m in z.namelist()
                if m.lower().endswith(".csv")
            ]

            if len(csv_members) != 1:

                raise ValueError(
                    f"{file.name} contains "
                    f"{len(csv_members)} CSV files"
                )

            with z.open(csv_members[0]) as f:

                return pd.read_csv(
                    f,
                    **kwargs
                )

    return pd.read_csv(
        file,
        compression="infer",
        **kwargs
    )


# ---------------------------------------------------------
# Discover Complete VMs
# ---------------------------------------------------------

def discover_complete_vm_ids(files):

    counts = {}

    for file in files:

        log.info(
            f"Scanning completeness: {file.name}"
        )

        df = read_csv_file(
            file,
            header=None,
            names=SCHEMA,
            usecols=[0],
            dtype={
                "encrypted_vmid": "string"
            }
        )

        vc = (
            df["encrypted_vmid"]
            .value_counts()
        )

        for vmid, cnt in vc.items():

            counts[vmid] = (
                counts.get(vmid, 0)
                + int(cnt)
            )

    complete_vms = [
        vmid
        for vmid, cnt in counts.items()
        if cnt == EXPECTED_POINTS
    ]

    mapping = pd.DataFrame({
        "encrypted_vmid": sorted(
            complete_vms
        )
    })

    log.info(
        f"Found {len(mapping):,} complete VMs "
        f"with {EXPECTED_POINTS} datapoints"
    )

    return mapping


# ---------------------------------------------------------
# Aggregate
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

    for idx, file in enumerate(files, 1):

        log.info(
            f"[{idx}/{len(files)}] Processing "
            f"{file.name}"
        )

        df = read_csv_file(
            file,
            header=None,
            names=SCHEMA,
            dtype=DTYPES
        )

        total_rows += len(df)

        df = df[
            df["encrypted_vmid"].isin(keep)
        ].copy()

        df["vm_id"] = (
            df["encrypted_vmid"]
            .map(lookup)
            .astype(np.int32)
        )

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

    hourly = pd.concat(
        result,
        ignore_index=True
    )

    hourly.sort_values(
        ["vm_id", "hour_index"],
        inplace=True
    )

    hourly.reset_index(
        drop=True,
        inplace=True
    )

    log.info(
        f"Processed {total_rows:,} raw rows"
    )

    return hourly


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

def write_report(df, outfile):

    hours = df.groupby("vm_id").size()

    lines = []

    lines.append(
        f"VMs               : {df.vm_id.nunique():,}"
    )

    lines.append(
        f"Rows              : {len(df):,}"
    )

    lines.append(
        f"Min hours / VM    : {hours.min()}"
    )

    lines.append(
        f"Median hours / VM : {int(hours.median())}"
    )

    lines.append(
        f"Max hours / VM    : {hours.max()}"
    )

    outfile.write_text(
        "\n".join(lines)
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    args = parse_args()

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    files = find_files(
        args.input
    )

    mapping = discover_complete_vm_ids(
        files
    )

    eligible_vms = len(mapping)

    if eligible_vms < TARGET_VM_COUNT:

        raise ValueError(
            f"Found only "
            f"{eligible_vms:,} complete VMs. "
            f"Need {TARGET_VM_COUNT:,}."
        )

    mapping = (
        mapping
        .sample(
            n=TARGET_VM_COUNT,
            random_state=RANDOM_SEED
        )
        .sort_values(
            "encrypted_vmid"
        )
        .reset_index(drop=True)
    )

    mapping["vm_id"] = np.arange(
        TARGET_VM_COUNT,
        dtype=np.int32
    )

    log.info(
        f"Selected "
        f"{TARGET_VM_COUNT:,} VMs "
        f"from "
        f"{eligible_vms:,} eligible VMs"
    )

    mapping.to_parquet(
        args.output /
        "vm_id_mapping.parquet",
        index=False
    )

    hourly = aggregate(
        files,
        mapping
    )

    parquet_file = (
        args.output /
        f"hourly_n{TARGET_VM_COUNT}.parquet"
    )

    csv_file = (
        args.output /
        f"hourly_n{TARGET_VM_COUNT}.csv"
    )

    hourly.to_parquet(
        parquet_file,
        index=False,
        compression="snappy"
    )

    hourly.to_csv(
        csv_file,
        index=False
    )

    write_report(
        hourly,
        args.output /
        "aggregation_report.txt"
    )

    log.info(
        f"Saved {parquet_file}"
    )

    log.info(
        f"Saved {csv_file}"
    )

    log.info(
        "Aggregation completed successfully"
    )


if __name__ == "__main__":
    main()