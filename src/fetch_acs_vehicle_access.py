#!/usr/bin/env python3
"""
Download tract-level ACS vehicle access data for Shelby County, Tennessee.

Usage:
    python fetch_acs_vehicle_access.py
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

from project_paths import PRESENTATION_READY_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

ACS_BASE_URL = "https://api.census.gov/data"
SHELBY_STATE_FIPS = "47"
SHELBY_COUNTY_FIPS = "157"
DEFAULT_YEARS = [2022, 2023, 2024]

FULL_TRACT_MAP_PATH = PRESENTATION_READY_DIR / "shelby_full_tract_cluster_map.geojson"
RAW_OUTPUT_PATH = RAW_DATA_DIR / "external" / "acs_b08201_shelby_tract_2022_2024.parquet"
PROCESSED_OUTPUT_PATH = PROCESSED_DATA_DIR / "shelby_acs_vehicle_access_tract_2022_2024.parquet"

ACS_VAR_MAP = {
    "B08201_001E": "hh_total",
    "B08201_002E": "hh_no_vehicle",
    "B08201_003E": "hh_one_vehicle",
    "B08201_004E": "hh_two_vehicle",
    "B08201_005E": "hh_three_vehicle",
    "B08201_006E": "hh_four_plus_vehicle",
}

COUNT_COLUMNS = list(ACS_VAR_MAP.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Shelby County ACS B08201 tract vehicle-access data and save a processed parquet panel."
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


def normalize_geoid(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.extract(r"(\d{11})", expand=False)
        .fillna(values.astype("string"))
        .str.replace(r"\D", "", regex=True)
        .str.zfill(11)
    )


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
    tract_universe = (
        tracts_gdf[["geoid"]]
        .drop_duplicates(subset=["geoid"])
        .sort_values("geoid")
        .reset_index(drop=True)
    )
    return tract_universe


def fetch_acs_vehicle_year(year: int, api_key: str = "") -> pd.DataFrame:
    url = f"{ACS_BASE_URL}/{year}/acs/acs5"
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
            f"Failed to fetch ACS B08201 data for year={year} from {url}. Error: {exc}"
        ) from exc

    if api_key and "Invalid Key" in response.text[:1000]:
        print(f"CENSUS_API_KEY was rejected for {year}; retrying without a key.")
        return fetch_acs_vehicle_year(year=year, api_key="")

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

    for column in COUNT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].where(frame[column] >= 0)

    frame["year"] = int(year)
    frame["acs_period"] = build_acs_period(year)
    frame["acs_table"] = "B08201"
    frame["acs_dataset"] = "acs5"
    frame["state"] = frame["state"].astype(str).str.zfill(2)
    frame["county"] = frame["county"].astype(str).str.zfill(3)
    frame["tract"] = frame["tract"].astype(str).str.zfill(6)
    frame["geoid"] = normalize_geoid(frame["state"] + frame["county"] + frame["tract"])

    frame["hh_three_plus_vehicle"] = frame["hh_three_vehicle"] + frame["hh_four_plus_vehicle"]
    frame["no_vehicle_share"] = safe_share(frame["hh_no_vehicle"], frame["hh_total"])
    frame["one_vehicle_share"] = safe_share(frame["hh_one_vehicle"], frame["hh_total"])
    frame["two_vehicle_share"] = safe_share(frame["hh_two_vehicle"], frame["hh_total"])
    frame["three_vehicle_share"] = safe_share(frame["hh_three_vehicle"], frame["hh_total"])
    frame["four_plus_vehicle_share"] = safe_share(frame["hh_four_plus_vehicle"], frame["hh_total"])
    frame["three_plus_vehicle_share"] = safe_share(frame["hh_three_plus_vehicle"], frame["hh_total"])

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
        "hh_total",
        "hh_no_vehicle",
        "hh_one_vehicle",
        "hh_two_vehicle",
        "hh_three_vehicle",
        "hh_four_plus_vehicle",
        "hh_three_plus_vehicle",
        "no_vehicle_share",
        "one_vehicle_share",
        "two_vehicle_share",
        "three_vehicle_share",
        "four_plus_vehicle_share",
        "three_plus_vehicle_share",
    ]
    return frame[ordered].copy()


def build_processed_panel(raw_df: pd.DataFrame, tract_universe: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    base = pd.MultiIndex.from_product(
        [tract_universe["geoid"].astype("string").tolist(), sorted(years)],
        names=["geoid", "year"],
    ).to_frame(index=False)
    processed = base.merge(tract_universe, on="geoid", how="left", validate="many_to_one")
    processed = processed.merge(raw_df, on=["geoid", "year"], how="left", validate="one_to_one")
    processed["acs_match_found"] = processed["hh_total"].notna()
    processed["acs_period"] = processed["acs_period"].fillna(processed["year"].map(build_acs_period))
    processed["acs_table"] = processed["acs_table"].fillna("B08201")
    processed["acs_dataset"] = processed["acs_dataset"].fillna("acs5")
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
        print(
            f"{year} ({build_acs_period(int(year))} ACS 5-year): "
            f"{matched}/{expected_tracts} tracts have B08201 data."
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
        print(f"Fetching Shelby ACS vehicle-access data for {year}...")
        raw_frames.append(fetch_acs_vehicle_year(year=year, api_key=api_key))
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
    print(
        f"Rows: {len(processed_df):,} "
        f"({processed_df['geoid'].nunique()} tracts x {processed_df['year'].nunique()} years)"
    )
    print_coverage_summary(processed_df)


if __name__ == "__main__":
    main()
