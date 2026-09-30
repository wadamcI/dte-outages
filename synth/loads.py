"""Building load profiles from NREL ResStock / ComStock (End-Use Load Profiles, AMY2018).

As in SMART-DS, each synthetic building gets the 15-minute electricity profile of
a real ResStock (homes) or ComStock (commercial) building model from the same
county, scaled by floor area. A small sample of profiles is shared across the
feeder; each is written as a per-unit loadshape (peak = 1.0).

Data: https://data.openei.org/submissions/4520 (CC BY 4.0). Files are cached in
synth/cache/eulp/.
"""
from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024"
DATASETS = {
    "res": "resstock_amy2018_release_2",
    "com": "comstock_amy2018_release_2",
}
ELEC_KWH = "out.electricity.total.energy_consumption"
SQFT_TO_M2 = 0.092903
STEPS = 35040  # 15-minute steps in a year


@dataclass
class Profile:
    kind: str           # "res" | "com"
    bldg_id: int
    building_type: str
    floor_m2: float
    peak_kw: float
    annual_kwh: float
    pu: np.ndarray      # 35040 values, max 1.0

    @property
    def name(self) -> str:
        return f"{self.kind}_kw_{self.bldg_id}_pu"


def _get(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def metadata(kind: str, state: str, county: str, cache: Path) -> pd.DataFrame:
    """Baseline metadata. ResStock publishes it per state, ComStock per county."""
    ds = DATASETS[kind]
    if kind == "res":
        name = f"{state}_baseline_metadata_and_annual_results.parquet"
        url = f"{BASE}/{ds}/metadata_and_annual_results/by_state/state={state}/parquet/{name}"
    else:
        name = f"{state}_{county}_baseline.parquet"
        url = (f"{BASE}/{ds}/metadata_and_annual_results/by_state_and_county/full/parquet/"
               f"state={state}/county={county}/{name}")
    meta = pd.read_parquet(_get(url, cache / ds / name))
    return meta.reset_index() if meta.index.name == "bldg_id" else meta  # ResStock indexes by bldg_id


def _county_col(meta: pd.DataFrame) -> str:
    for col in ("in.county", "in.nhgis_county_gisjoin", "in.county_gisjoin"):
        if col in meta:
            return col
    raise KeyError("no county column in EULP metadata")


def _type_col(meta: pd.DataFrame, kind: str) -> str:
    for col in (("in.geometry_building_type_recs",) if kind == "res" else
                ("in.comstock_building_type", "in.building_type")):
        if col in meta:
            return col
    raise KeyError("no building-type column in EULP metadata")


def _floor_m2(row: pd.Series) -> float:
    for col in ("in.sqft", "in.sqft..ft2", "in.floor_area"):
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col]) * SQFT_TO_M2
            except (TypeError, ValueError):
                continue
    return float("nan")


def sample_profiles(kind: str, state: str, county: str, n: int, cache: Path,
                    types: list[str] | None = None, seed: int = 0) -> list[Profile]:
    """Download n building profiles of the given kind from one county."""
    meta = metadata(kind, state, county, cache)
    cc, tc = _county_col(meta), _type_col(meta, kind)
    pool = meta[meta[cc] == county]
    if types:
        typed = pool[pool[tc].isin(types)]
        pool = typed if len(typed) >= n else pool
    if pool.empty:
        raise ValueError(f"no {kind} buildings for county {county}")
    pick = pool.sample(min(n, len(pool)), random_state=seed)
    ds = DATASETS[kind]
    out = []
    for _, row in pick.iterrows():
        bid = int(row["bldg_id"])
        url = f"{BASE}/{ds}/timeseries_individual_buildings/by_state/upgrade=0/state={state}/{bid}-0.parquet"
        path = _get(url, cache / ds / "timeseries" / f"{bid}-0.parquet")
        ts = pd.read_parquet(path, columns=[ELEC_KWH])[ELEC_KWH].to_numpy(dtype=float)
        if len(ts) != STEPS:
            print(f"  skip {kind} {bid}: {len(ts)} steps", file=sys.stderr)
            continue
        kw = ts * 4  # kWh per 15 min -> kW
        out.append(Profile(kind, bid, str(row[tc]), _floor_m2(row), float(kw.max()),
                           float(ts.sum()), kw / kw.max()))
    print(f"  {kind}: {len(out)} profiles from {county}", file=sys.stderr)
    return out


def write_profile_csv(profile: Profile, directory: Path) -> Path:
    path = directory / f"{profile.name}.csv"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        np.savetxt(buf, profile.pu, fmt="%.5f")
        path.write_text(buf.getvalue())
    return path
