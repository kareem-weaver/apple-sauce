#!/usr/bin/env python3
"""
Download tract-level ACS internet access data for Shelby County, Tennessee.

Usage:
    python fetch_acs_internet_access.py
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from generate_shelby_black_population_overlay import FULL_TRACT_MAP_PATH, normalize_geoid
from project_paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

ACS_BASE_URL = "https://api.census.gov/data"
SHELBY_STATE_FIPS = "47"
SHELBY_COUNTY_FIPS = "157"
DEFAULT_YEARS = [2022, 2023, 2024]

RAW_OUTPUT_PATH = RAW_DATA_DIR / "external" / "acs_s2801_shelby_tract_2022_2024.parquet"
PROCESSED_OUTPUT_PATH = PROCESSED_DATA_DIR / "shelby_acs_internet_access_tract_2022_2024.parquet"

ACS_VAR_MAP = {
    "S2801_C01_001E": "total_households",
    "S2801_C01_001M": "total_households_moe",
    "S2801_C01_012E": "internet_subscription_households",
    "S2801_C01_012M": "internet_subscription_households_moe",
    "S2801_C02_012E": "internet_subscription_pct",
    "S2801_C02_012M": "internet_subscription_pct_moe",
    "S2801_C01_014E": "broadband_households",
    "S2801_C01_014M": "broadband_households_moe",
    "S2801_C02_014E": "broadband_pct",
    "S2801_C02_014M": "broadband_pct_moe",
    "S2801_C01_016E": "cellular_only_households",
    "S2801_C01_016M": "cellular_only_households_moe",
    "S2801_C02_016E": "cellular_only_pct",
    "S2801_C02_016M": "cellular_only_pct_moe",
    "S2801_C01_019E": "no_internet_households",
    "S2801_C01_019M": "no_internet_households_moe",
    "S2801_C02_019E": "no_internet_pct",
    "S2801_C02_019M": "no_internet_pct_moe",
}

COUNT_COLUMNS = [
    "total_households",
    "total_households_moe",
    "internet_subscription_households",
    "internet_subscription_households_moe",
    "broadband_households",
    "broadband_households_moe",
    "cellular_only_households",
    "cellular_only_households_moe",
    "no_internet_households",
    "no_internet_households_moe",
]

PCT_COLUMNS = [
    "internet_subscription_pct",
    "internet_subscription_pct_moe",
    "broadband_pct",
    "broadband_pct_moe",
    "cellular_only_pct",
    "cellular_only_pct_moe",
    "no_internet_pct",
    "no_internet_pct_moe",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Shelby County ACS S2801 tract internet access data and save a processed parquet panel."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="ACS release years to fetch. Defaults to 2022 2023 2024.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_OUTPUT_PATH,
        help="Processed parquet output path.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=RAW_OUTPUT_PATH,
        help="Raw parquet cache output path.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.20,
        help="Pause between Census API requests.",
    )
    return parser.parse_args()


def build_acs_period(year: int) -> str:
    return f"{year - 4}-{year}"


def safe_share(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = pd.Series(np.nan, index=num.index, dtype=float)
    valid = num.notna() & den.notna() & (den > 0)
    out.loc[valid] = num.loc[valid] / den.loc[valid]
    return out


def load_current_shelby_tract_universe() -> pd.DataFrame:
    tracts_gdf = gpd.read_file(FULL_TRACT_MAP_PATH)
    tracts_gdf["geoid"] = normalize_geoid(tracts_gdf["geoid"])
    tracts_gdf = tracts_gdf.dropna(subset=["geoid"]).copy()
    tracts_gdf["geoid"] = tracts_gdf["geoid"].astype("string")
    tract_cols = ["geoid"]
    if "NAMELSAD" in tracts_gdf.columns:
        tract_cols.append("NAMELSAD")
    tract_universe = (
        tracts_gdf[tract_cols]
        .drop_duplicates(subset=["geoid"])
        .sort_values("geoid")
        .reset_index(drop=True)
    )
    if "NAMELSAD" in tract_universe.columns:
        tract_universe = tract_universe.rename(columns={"NAMELSAD": "tract_name"})
    return tract_universe


def fetch_acs_internet_year(year: int, api_key: str = "") -> pd.DataFrame:
    url = f"{ACS_BASE_URL}/{year}/acs/acs5/subject"
    params = {
        "get": ",".join(["NAME"] + list(ACS_VAR_MAP.keys())),
        "for": "tract:*",
        "in": f"state:{SHELBY_STATE_FIPS} county:{SHELBY_COUNTY_FIPS}",
    }
    if api_key:
        params["key"] = api_key

    try:
        response = requests.get(url, params=params, timeout=(20, 120))
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch ACS S2801 data for year={year} from {url}. "
            f"Error: {exc}"
        ) from exc

    if api_key and "Invalid Key" in response.text[:1000]:
        print(f"CENSUS_API_KEY was rejected for {year}; retrying without a key.")
        return fetch_acs_internet_year(year=year, api_key="")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Census API did not return JSON for year={year}. Response preview: {response.text[:500]}"
        ) from exc

    if not data or len(data) < 2:
        raise RuntimeError(f"Census API returned no tract rows for year={year}.")

    frame = pd.DataFrame(data[1:], columns=data[0]).rename(columns=ACS_VAR_MAP).copy()
    frame = frame.rename(columns={"NAME": "name"})

    for column in COUNT_COLUMNS + PCT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].where(frame[column] >= 0)

    frame["year"] = int(year)
    frame["acs_period"] = build_acs_period(year)
    frame["acs_table"] = "S2801"
    frame["acs_dataset"] = "acs5/subject"
    frame["state"] = frame["state"].astype(str).str.zfill(2)
    frame["county"] = frame["county"].astype(str).str.zfill(3)
    frame["tract"] = frame["tract"].astype(str).str.zfill(6)
    frame["geoid"] = normalize_geoid(frame["state"] + frame["county"] + frame["tract"])

    frame["internet_sub_share"] = pd.to_numeric(frame["internet_subscription_pct"], errors="coerce") / 100.0
    frame["internet_sub_share_from_counts"] = safe_share(
        frame["internet_subscription_households"], frame["total_households"]
    )
    frame["internet_sub_share"] = frame["internet_sub_share"].fillna(frame["internet_sub_share_from_counts"])
    frame["broadband_share"] = pd.to_numeric(frame["broadband_pct"], errors="coerce") / 100.0
    frame["broadband_share_from_counts"] = safe_share(
        frame["broadband_households"], frame["total_households"]
    )
    frame["broadband_share"] = frame["broadband_share"].fillna(frame["broadband_share_from_counts"])
    frame["cellular_only_share"] = pd.to_numeric(frame["cellular_only_pct"], errors="coerce") / 100.0
    frame["cellular_only_share_from_counts"] = safe_share(
        frame["cellular_only_households"], frame["total_households"]
    )
    frame["cellular_only_share"] = frame["cellular_only_share"].fillna(frame["cellular_only_share_from_counts"])
    frame["no_internet_share"] = pd.to_numeric(frame["no_internet_pct"], errors="coerce") / 100.0
    frame["no_internet_share_from_counts"] = safe_share(
        frame["no_internet_households"], frame["total_households"]
    )
    frame["no_internet_share"] = frame["no_internet_share"].fillna(frame["no_internet_share_from_counts"])
    frame["has_households"] = frame["total_households"].fillna(0) > 0

    ordered = [
        "geoid",
        "state",
        "county",
        "tract",
        "name",
        "year",
        "acs_period",
        "acs_table",
        "acs_dataset",
        "has_households",
        "total_households",
        "total_households_moe",
        "internet_subscription_households",
        "internet_subscription_households_moe",
        "internet_subscription_pct",
        "internet_subscription_pct_moe",
        "internet_sub_share",
        "internet_sub_share_from_counts",
        "broadband_households",
        "broadband_households_moe",
        "broadband_pct",
        "broadband_pct_moe",
        "broadband_share",
        "broadband_share_from_counts",
        "cellular_only_households",
        "cellular_only_households_moe",
        "cellular_only_pct",
        "cellular_only_pct_moe",
        "cellular_only_share",
        "cellular_only_share_from_counts",
        "no_internet_households",
        "no_internet_households_moe",
        "no_internet_pct",
        "no_internet_pct_moe",
        "no_internet_share",
        "no_internet_share_from_counts",
    ]
    return frame[ordered].copy()


def build_processed_panel(raw_df: pd.DataFrame, tract_universe: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    base = pd.MultiIndex.from_product(
        [tract_universe["geoid"].astype("string").tolist(), sorted(years)],
        names=["geoid", "year"],
    ).to_frame(index=False)
    processed = base.merge(tract_universe, on="geoid", how="left", validate="many_to_one")
    processed = processed.merge(raw_df, on=["geoid", "year"], how="left", validate="one_to_one")
    processed["acs_match_found"] = processed["internet_sub_share"].notna()
    processed["acs_period"] = processed["acs_period"].fillna(processed["year"].map(build_acs_period))
    processed["acs_table"] = processed["acs_table"].fillna("S2801")
    processed["acs_dataset"] = processed["acs_dataset"].fillna("acs5/subject")
    return processed


def print_coverage_summary(processed_df: pd.DataFrame) -> None:
    expected_tracts = processed_df["geoid"].nunique()
    print()
    print("Coverage summary")
    print("----------------")
    for year in sorted(processed_df["year"].dropna().unique()):
        sub = processed_df.loc[processed_df["year"] == year].copy()
        matched = int(sub["acs_match_found"].sum())
        missing = expected_tracts - matched
        share_missing = int(sub["internet_sub_share"].isna().sum())
        zero_household = int(sub["has_households"].fillna(False).eq(False).sum())
        print(
            f"{year} ({build_acs_period(int(year))} ACS 5-year): "
            f"{matched}/{expected_tracts} tracts have usable internet-subscription shares; "
            f"{share_missing} rows are missing internet_sub_share; "
            f"{zero_household} tracts report zero households."
        )
        if missing > 0:
            sample = sub.loc[~sub["acs_match_found"], "geoid"].astype(str).head(5).tolist()
            if sample:
                print(f"  Sample unmatched GEOIDs: {sample}")


def main() -> None:
    args = parse_args()
    years = sorted({int(year) for year in args.years})
    api_key = os.getenv("CENSUS_API_KEY", "").strip()

    raw_frames: list[pd.DataFrame] = []
    for year in years:
        print(f"Fetching Shelby ACS internet access data for {year}...")
        raw_frames.append(fetch_acs_internet_year(year=year, api_key=api_key))
        time.sleep(max(args.pause_seconds, 0.0))

    raw_df = pd.concat(raw_frames, ignore_index=True)
    raw_df = raw_df.sort_values(["year", "geoid"]).reset_index(drop=True)

    tract_universe = load_current_shelby_tract_universe()
    processed_df = build_processed_panel(raw_df=raw_df, tract_universe=tract_universe, years=years)
    processed_df = processed_df.sort_values(["year", "geoid"]).reset_index(drop=True)

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_parquet(args.raw_output, index=False)
    processed_df.to_parquet(args.output, index=False)

    print(f"Saved raw ACS cache: {args.raw_output}")
    print(f"Saved processed Shelby tract panel: {args.output}")
    print(f"Rows: {len(processed_df):,} ({processed_df['geoid'].nunique()} tracts x {processed_df['year'].nunique()} years)")
    print_coverage_summary(processed_df)


if __name__ == "__main__":
    main()
