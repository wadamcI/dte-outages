#!/usr/bin/env python3
"""Build a SMART-DS-style synthetic distribution feeder, constrained by DTE outage data.

SMART-DS (NREL) generates feeders from buildings, streets and ResStock/ComStock
loads with a planning model. This does the same at a smaller scale and adds what
a year of outages reveals about the real circuit:

  observed transformer locations     octagon centres of transformer outages
  customers per observed transformer max NUM_CUST of those outages
  protective-device service areas    Device (I)/(II) outage footprints, which must
                                     each be a subtree fed through one device

Pipeline:
  1. Buildings (Overture) inside the circuit footprint -> service points with a
     ResStock/ComStock profile scaled by floor area.
  2. Observed transformers take their observed number of nearby buildings; the
     rest are clustered into synthetic transformers (same median size).
  3. Primary: approximate Steiner tree over OSM roads from the substation to every
     transformer. Outage-constrained: each device footprint is wired first as its
     own subtree, then joined to the trunk.
  4. Devices placed where each observed footprint's transformers branch off;
     3-phase where downstream load is large, 1-phase laterals phase-balanced.
  5. OpenDSS files in the SMART-DS layout, a peak power flow, and a held-out
     check of how well the tree reproduces device footprints (vs. no constraint).

    python3 synth/build_feeder.py PON-C008
Outputs: synth/out/<CIRCUIT>/opendss/..., network.html, build_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pydeck as pdk
from networkx.algorithms.approximation import steiner_tree
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loads as eulp  # noqa: E402
from pilot_circuit import (CACHE, METRIC, OUT, ROOT, circuit_features, footprints_for,  # noqa: E402
                           osm, osm_frames, overture_buildings, primary_buildings)

SNAP_M = 25                 # road segments are split into pieces this long
AVOID_FACTOR = 50           # cost multiplier that keeps the trunk out of an already-wired device area
THREE_PHASE_KW = 300        # downstream coincident peak above which the primary is 3-phase
MULTIFAMILY_M2 = 400        # untagged buildings this large: multifamily (ResStock x units, 3-phase)
COMMERCIAL_M2 = 1500        # untagged buildings this large: commercial (ComStock, 3-phase)
M2_PER_UNIT = 90            # floor area per apartment when estimating multifamily units
COMMERCIAL_TAGS = ["commercial", "industrial", "civic", "education", "religious", "retail", "medical", "service"]
# Rough thermal capacity of one DTE circuit by voltage (kVA); used to choose a feasible substation.
CIRCUIT_KVA = {4.8: 4000, 8.32: 7000, 13.2: 12000}
DIVERSITY = 0.6             # coincidence factor when sizing shared transformers
# Street-aware transformer placement
REACH_ALONG_M = 60          # default reach of one transformer; replaced by the reach measured from outages
REAR_LOT_SETBACK_M = 20     # observed transformers further than this behind the street -> rear-lot layout
MAX_OBS_DROP_M = 100        # observed transformers only take houses within this distance
OWN_TX_SETBACK_M = 90       # houses set back further than this get their own transformer near the house
TAP_MIN_M = 5               # transformers further than this from the primary get a tap line
LONG_DROP_M = 75            # service drops longer than this are unusual (reported)
NON_FRONTAGE_SERVICE = {"driveway", "parking_aisle", "drive-through", "emergency_access"}
# SMART-DS conductors by ampacity (synth/smartds_linecodes.dss), smallest adequate one is used
PRIMARY_3PH = [(325, "3P_OH_AL_ACSR_4/0_Penguin_3"), (595, "3P_OH_AL_ACSR_477kcmil_Hawk_3"),
               (960, "3P_OH_AL_ACSR_1033kcmil_Curlew_3")]
SWITCH_3PH = [(295, "padswitch_3"), (550, "padswitch_3_5"), (960, "padswitch_3_11")]
SERVICE_1PH = [(115, "1P_OH_AL_2/0_Rucina_2"), (301, "1P_UG_AL_4/0_Sweetbriar_2")]
SERVICE_3PH = [(370, "3P_UG_AL_350kcmil_3_3"), (550, "3P_UG_AL_750kcmil_3_2")]
SIZING_MARGIN = 1.25
TX_1PH_KVA = [10, 15, 25, 37.5, 50, 75, 100, 167, 250]
TX_3PH_KVA = [75, 112.5, 150, 225, 300, 500, 750, 1000, 1500, 2000]
N_RES_PROFILES, N_COM_PROFILES = 24, 8
COUNTY = {"PON": "G2601250", "RFD": "G2601630", "WWS": "G2601630", "ANN": "G2601610",
          "CAN": "G2600990", "MTC": "G2600990", "NPT": "G2601150", "SBY": "G2601470",
          "HWL": "G2600930", "LAP": "G2600870", "NAE": "G2600990"}  # service center -> county (approx.)


# ------------------------------------------------------------------ helpers

def std_size(kva: float, sizes: list[float]) -> float:
    return next((s for s in sizes if s >= kva), sizes[-1])


def conductor(amps: float, table: list[tuple[float, str]]) -> tuple[str, int]:
    """Smallest adequate conductor, or parallel runs of the largest one."""
    for rating, code in table:
        if amps <= rating:
            return code, 1
    rating, code = table[-1]
    return code, math.ceil(amps / rating)


def bus(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name).lower()


def lonlat(points: gpd.GeoSeries) -> np.ndarray:
    ll = points.to_crs("EPSG:4326")
    return np.column_stack([ll.x, ll.y])


# ----------------------------------------------------------------- road graph

def road_graph(roads: gpd.GeoDataFrame, area) -> nx.Graph:
    """Undirected graph of road pieces <= SNAP_M long (excluding freeways)."""
    r = roads[~roads["highway"].fillna("").str.startswith("motorway")]
    r = r[r.intersects(area)]
    g = nx.Graph()
    for geom in r.geometry:
        if geom.geom_type != "LineString":
            continue
        coords = list(geom.segmentize(SNAP_M).coords)
        for a, b in zip(coords, coords[1:]):
            a, b = (round(a[0], 1), round(a[1], 1)), (round(b[0], 1), round(b[1], 1))
            if a != b:
                g.add_edge(a, b, weight=math.dist(a, b), length=math.dist(a, b))
    return g.subgraph(max(nx.connected_components(g), key=len)).copy()


class Snapper:
    def __init__(self, g: nx.Graph):
        self.nodes = list(g.nodes)
        self.tree = KDTree(np.array(self.nodes))

    def __call__(self, xy) -> tuple[tuple, float]:
        d, i = self.tree.query(np.atleast_2d(xy), k=1)
        return self.nodes[int(i[0][0])], float(d[0][0])


# ---------------------------------------------------------------- transformers

def assign_transformers(bld: gpd.GeoDataFrame, tx_obs: gpd.GeoDataFrame, bldg_per_cust: float,
                        median_tx_customers: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Observed transformers take their customer count in nearest buildings; the
    rest of the residential buildings are clustered into synthetic transformers."""
    res = bld[~bld["commercial"]]
    xy = np.column_stack([res.geometry.centroid.x, res.geometry.centroid.y])
    owner = np.full(len(res), -1)
    txs = []
    tree = KDTree(xy)
    for t in tx_obs.sort_values("customers", ascending=False).itertuples():
        want = max(1, round(t.customers * bldg_per_cust))
        c = t.geometry.centroid
        dist, idx = tree.query([[c.x, c.y]], k=min(len(xy), want * 4 + 10))
        take = [i for d, i in zip(dist[0], idx[0]) if owner[i] < 0 and d <= 150][:want]
        tid = len(txs)
        owner[take] = tid
        txs.append({"tx": tid, "x": c.x, "y": c.y, "observed": True, "phases": 1,
                    "obs_customers": int(t.customers), "footprint": t.id, "placement": "observed"})
    rest = np.flatnonzero(owner < 0)
    per_tx = max(4, round(median_tx_customers * bldg_per_cust))
    if len(rest):
        k = max(1, math.ceil(len(rest) / per_tx))
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(xy[rest])
        for lab in range(k):
            members = rest[km.labels_ == lab]
            if not len(members):
                continue
            tid = len(txs)
            owner[members] = tid
            cx, cy = xy[members].mean(axis=0)
            txs.append({"tx": tid, "x": cx, "y": cy, "observed": False, "phases": 1,
                        "obs_customers": None, "footprint": None, "placement": "kmeans"})
    # commercial buildings each get their own 3-phase pad-mount transformer
    full_owner = np.full(len(bld), -1)
    full_owner[np.flatnonzero(~bld["commercial"].to_numpy())] = owner
    for i in np.flatnonzero(bld["commercial"].to_numpy()):
        c = bld.geometry.iloc[i].centroid
        tid = len(txs)
        full_owner[i] = tid
        txs.append({"tx": tid, "x": c.x, "y": c.y, "observed": False, "phases": 3,
                    "obs_customers": None, "footprint": None, "placement": "building"})
    return pd.DataFrame(txs), full_owner


def frontage_roads(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Streets and alleys a house can be served from (not freeways, driveways or parking aisles)."""
    hw, svc = roads["highway"].fillna(""), roads["service"].fillna("")
    ok = ~hw.str.startswith("motorway") & ~((hw == "service") & (svc != "alley")) & ~svc.isin(NON_FRONTAGE_SERVICE)
    return roads[ok & (roads.geometry.geom_type == "LineString")].reset_index(drop=True)


def assign_transformers_street(bld: gpd.GeoDataFrame, tx_obs: gpd.GeoDataFrame, bldg_per_cust: float,
                               median_tx_customers: float, fr: gpd.GeoDataFrame,
                               reach: float = REACH_ALONG_M) -> tuple[pd.DataFrame, np.ndarray]:
    """Street-aware placement, as utilities lay out secondaries.

    1. Observed transformers (outage octagon centres) take their observed number of
       nearest houses, within MAX_OBS_DROP_M.
    2. Every other house is attached to its frontage street or alley (the nearest
       one), and the houses along each street are cut into contiguous runs: a new
       transformer whenever a run would exceed the observed transformer size or
       span more than 2 x reach. Each transformer sits on the street at the
       middle of its run, where a pole would be, so no drop crosses another street.
    3. Houses set back beyond OWN_TX_SETBACK_M (long rural driveways) get their own
       transformer near the house; commercial/multifamily get a pad-mount."""
    cent = bld.geometry.centroid
    owner = np.full(len(bld), -1)
    txs: list[dict] = []
    res_idx = np.flatnonzero(~bld["commercial"].to_numpy())
    xy = np.column_stack([cent.x.to_numpy()[res_idx], cent.y.to_numpy()[res_idx]])
    tree = KDTree(xy)
    for t in tx_obs.sort_values("customers", ascending=False).itertuples():
        want = max(1, round(t.customers * bldg_per_cust))
        c = t.geometry.centroid
        dist, idx = tree.query([[c.x, c.y]], k=min(len(xy), want * 4 + 10))
        take = [res_idx[i] for d, i in zip(dist[0], idx[0]) if owner[res_idx[i]] < 0 and d <= MAX_OBS_DROP_M][:want]
        owner[take] = len(txs)
        txs.append({"tx": len(txs), "x": c.x, "y": c.y, "observed": True, "phases": 1,
                    "obs_customers": int(t.customers), "footprint": t.id, "placement": "observed"})

    rest = res_idx[owner[res_idx] < 0]
    per_tx = max(4, round(median_tx_customers * bldg_per_cust))
    near = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=cent.iloc[rest].to_numpy(), crs=METRIC),
                             fr[["geometry"]], how="left", distance_col="setback")
    near = near[~near.index.duplicated()]
    by_road: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for b, r, setback in zip(rest, near["index_right"].to_numpy(), near["setback"].to_numpy()):
        if setback > OWN_TX_SETBACK_M:
            c = cent.iloc[b]
            owner[b] = len(txs)
            txs.append({"tx": len(txs), "x": c.x, "y": c.y, "observed": False, "phases": 1,
                        "obs_customers": None, "footprint": None, "placement": "setback"})
            continue
        by_road[int(r)].append((fr.geometry.iloc[int(r)].project(cent.iloc[b]), int(b)))
    for r, items in by_road.items():
        line = fr.geometry.iloc[r]
        items.sort()
        pos = np.array([p for p, _ in items])
        ids = np.array([b for _, b in items])
        breaks = np.flatnonzero(np.diff(pos) > 2 * reach) + 1   # gaps split a street into runs
        for run_pos, run_ids in zip(np.split(pos, breaks), np.split(ids, breaks)):
            k = max(1, math.ceil(len(run_ids) / per_tx), math.ceil((run_pos[-1] - run_pos[0]) / (2 * reach)))
            for chunk_pos, chunk_ids in zip(np.array_split(run_pos, k), np.array_split(run_ids, k)):
                if not len(chunk_ids):
                    continue
                pole = line.interpolate(float(np.median(chunk_pos)))
                owner[chunk_ids] = len(txs)
                txs.append({"tx": len(txs), "x": pole.x, "y": pole.y, "observed": False, "phases": 1,
                            "obs_customers": None, "footprint": None, "placement": "street"})

    merge_small_groups(txs, owner, cent, per_tx)

    for i in np.flatnonzero(bld["commercial"].to_numpy()):
        c = cent.iloc[i]
        owner[i] = len(txs)
        txs.append({"tx": len(txs), "x": c.x, "y": c.y, "observed": False, "phases": 3,
                    "obs_customers": None, "footprint": None, "placement": "building"})
    return pd.DataFrame(txs), owner


def observed_layout(tx_obs: gpd.GeoDataFrame, bld: gpd.GeoDataFrame, bldg_per_cust: float,
                    fr: gpd.GeoDataFrame) -> dict:
    """What the outage-located transformers say about this circuit's secondaries:
    how far a transformer reaches (distance to the farthest of its N nearest houses,
    N = its observed customers; 75th percentile) and how far they sit from the street."""
    res = bld[~bld["commercial"]]
    if len(tx_obs) < 3 or len(res) < 10:
        return {"n": len(tx_obs), "reach_m": float(REACH_ALONG_M), "setback_m": 0.0}
    xy = np.column_stack([res.geometry.centroid.x, res.geometry.centroid.y])
    tree = KDTree(xy)
    far = []
    for t in tx_obs.itertuples():
        c = t.geometry.centroid
        d, _ = tree.query([[c.x, c.y]], k=min(len(xy), max(1, round(t.customers * bldg_per_cust))))
        far.append(d[0].max())
    cent = gpd.GeoDataFrame(geometry=tx_obs.geometry.centroid.to_numpy(), crs=METRIC)
    setback = gpd.sjoin_nearest(cent, fr[["geometry"]], distance_col="d").groupby(level=0)["d"].min()
    # 75th percentile: a hard per-transformer limit at the median would split half of real ones
    return {"n": len(tx_obs), "reach_m": round(float(np.clip(np.quantile(far, 0.75), 40, 120)), 1),
            "setback_m": round(float(setback.median()), 1)}


def assign_transformers_block(bld: gpd.GeoDataFrame, tx_obs: gpd.GeoDataFrame, bldg_per_cust: float,
                              median_tx_customers: float, fr: gpd.GeoDataFrame,
                              reach: float, merge_blocks: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Rear-lot layout: houses are grouped by city block (faces of the street network),
    so no service drop crosses a street, then clustered within each block to the
    observed transformer size, splitting any group whose farthest house is beyond the
    measured reach. The centre of back-to-back houses is their shared rear lot line."""
    cent = bld.geometry.centroid
    owner = np.full(len(bld), -1)
    txs: list[dict] = []
    res_idx = np.flatnonzero(~bld["commercial"].to_numpy())
    xy_all = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    tree = KDTree(xy_all[res_idx])
    for t in tx_obs.sort_values("customers", ascending=False).itertuples():
        want = max(1, round(t.customers * bldg_per_cust))
        c = t.geometry.centroid
        dist, idx = tree.query([[c.x, c.y]], k=min(len(res_idx), want * 4 + 10))
        take = [res_idx[i] for d, i in zip(dist[0], idx[0]) if owner[res_idx[i]] < 0 and d <= MAX_OBS_DROP_M][:want]
        owner[take] = len(txs)
        txs.append({"tx": len(txs), "x": c.x, "y": c.y, "observed": True, "phases": 1,
                    "obs_customers": int(t.customers), "footprint": t.id, "placement": "observed"})

    rest = res_idx[owner[res_idx] < 0]
    blocks = gpd.GeoDataFrame(geometry=list(polygonize(unary_union(list(fr.geometry)))), crs=METRIC)
    pts = gpd.GeoDataFrame(geometry=cent.iloc[rest].to_numpy(), crs=METRIC)
    j = gpd.sjoin_nearest(pts, blocks[["geometry"]], how="left", max_distance=50)
    j = j[~j.index.duplicated()]
    block_of = j["index_right"].fillna(-1).astype(int).to_numpy()   # -1: not inside any block
    per_tx = max(4, round(median_tx_customers * bldg_per_cust))
    for b in np.unique(block_of):
        members = rest[block_of == b]
        xy = xy_all[members]
        k = max(1, math.ceil(len(members) / per_tx))
        labels = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(xy) if k > 1 else np.zeros(len(xy), int)
        groups = [members[labels == i] for i in range(k)]
        while groups:  # split only the groups with a drop beyond the observed reach
            m = groups.pop()
            if not len(m):
                continue
            centre = xy_all[m].mean(axis=0)
            if len(m) > 1 and np.hypot(*(xy_all[m] - centre).T).max() > reach:
                halves = KMeans(n_clusters=2, n_init=3, random_state=0).fit_predict(xy_all[m])
                groups += [m[halves == 0], m[halves == 1]]
                continue
            owner[m] = len(txs)
            txs.append({"tx": len(txs), "x": centre[0], "y": centre[1], "observed": False,
                        "phases": 1, "obs_customers": None, "footprint": None, "placement": "block"})
    if merge_blocks:  # tiny blocks: share a neighbouring block's transformer if within reach
        merge_small_groups(txs, owner, cent, per_tx, kind="block", max_drop=reach)

    for i in np.flatnonzero(bld["commercial"].to_numpy()):
        c = cent.iloc[i]
        owner[i] = len(txs)
        txs.append({"tx": len(txs), "x": c.x, "y": c.y, "observed": False, "phases": 3,
                    "obs_customers": None, "footprint": None, "placement": "building"})
    return pd.DataFrame(txs), owner


def merge_small_groups(txs: list[dict], owner: np.ndarray, cent: gpd.GeoSeries, per_tx: int,
                       kind: str = "street", max_drop: float = LONG_DROP_M) -> None:
    """Fold undersized street transformers into a neighbour (around a corner, or across
    the short pieces OSM splits streets into) when every moved drop stays short.
    Emptied transformers are dropped later (they serve no buildings)."""
    street = [t["tx"] for t in txs if t["placement"] == kind]
    if len(street) < 2:
        return
    poles = np.array([(txs[t]["x"], txs[t]["y"]) for t in street])
    tree = KDTree(poles)
    xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    size = {t: int((owner == t).sum()) for t in street}
    alive = set(street)
    for t in sorted(street, key=size.get):
        if t not in alive or size[t] >= per_tx / 2:
            continue
        members = np.flatnonzero(owner == t)
        _, idx = tree.query([poles[street.index(t)]], k=min(8, len(street)))
        best, best_drop = None, max_drop
        for c in (street[i] for i in idx[0]):
            if c == t or c not in alive or size[c] + len(members) > per_tx * 1.25:
                continue
            drop = float(np.hypot(*(xy[members] - (txs[c]["x"], txs[c]["y"])).T).max())
            if drop <= best_drop:
                best, best_drop = c, drop
        if best is not None:
            owner[members] = best
            size[best] += len(members)
            alive.discard(t)


def service_stats(bld: gpd.GeoDataFrame, txs: pd.DataFrame, fr: gpd.GeoDataFrame,
                  sites: gpd.GeoSeries) -> dict:
    """How realistic the secondaries are: drop lengths, drops crossing a street, and
    how far observed customer sites (single-customer outages) are from their transformer."""
    pos = dict(zip(txs["tx"], zip(txs["x"], txs["y"])))
    cent = bld.geometry.centroid
    res = ~bld["commercial"].to_numpy()
    drop = np.array([math.dist((c.x, c.y), pos[t]) for c, t in zip(cent, bld["tx"])])
    lines = gpd.GeoDataFrame(geometry=[LineString([(c.x, c.y), pos[t]]) for c, t in zip(cent, bld["tx"])], crs=METRIC)
    hits = gpd.sjoin(lines[res], fr[["geometry"]], predicate="intersects")
    crossing = {i for i, r in zip(hits.index, hits["index_right"])
                if lines.geometry.iloc[i].intersection(fr.geometry.iloc[r]).distance(Point(pos[bld["tx"].iloc[i]])) > 3}
    out = {"median_drop_m": round(float(np.median(drop[res])), 1),
           "p90_drop_m": round(float(np.quantile(drop[res], 0.9)), 1),
           f"share_drops_over_{LONG_DROP_M}m": round(float((drop[res] > LONG_DROP_M).mean()), 3),
           "share_drops_crossing_a_street": round(len(crossing) / max(1, int(res.sum())), 3)}
    if len(sites):
        near = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=sites.to_numpy(), crs=METRIC),
                                 gpd.GeoDataFrame({"tx": bld["tx"].to_numpy()}, geometry=cent.to_numpy(), crs=METRIC))
        near = near[~near.index.duplicated()]
        d = np.array([math.dist((p.x, p.y), pos[t]) for p, t in zip(sites, near["tx"])])
        out.update(observed_sites=len(sites), site_to_transformer_median_m=round(float(np.median(d)), 1),
                   **{f"sites_within_{LONG_DROP_M}m_of_transformer": round(float((d <= LONG_DROP_M).mean()), 3)})
    return out


# -------------------------------------------------------------------- primary

def build_tree(g: nx.Graph, head, tx_nodes: dict[int, tuple], dev_tx: dict[str, set[int]],
               devices: gpd.GeoDataFrame | None) -> nx.Graph:
    """Approximate Steiner tree over roads from the substation to every transformer.

    With `devices` (observed outage footprints), each device area is wired first as
    its own subtree, smallest first so nested devices compose. The subtree connects
    to the rest of the network through a single entry node (its node nearest the
    substation) and is made expensive to cross, so the trunk can't run through it.
    That is what makes each observed footprint a clean downstream branch."""
    work = g.copy()
    rep = {n: n for n in tx_nodes.values()}          # tx node -> current representative
    local_edges: list[tuple] = []
    if devices is not None and len(devices):
        to_head = nx.single_source_dijkstra_path_length(g, head, weight="length")
        for dev in devices.sort_values("area").itertuples():
            inside = [tx_nodes[t] for t in dev_tx.get(dev.id, ())]
            terms = {rep[n] for n in inside}
            if len(terms) < 2:
                continue
            zone = dev.geometry.buffer(60)
            sub = work.subgraph([n for n in work.nodes if zone.contains(Point(n)) or n in terms])
            for comp in (c for c in nx.connected_components(sub) if len(c & terms) >= 2):
                local = steiner_tree(sub.subgraph(comp), list(comp & terms), weight="weight", method="mehlhorn")
                entry = min(local.nodes, key=lambda n: to_head.get(n, math.inf))
                local_edges += list(local.edges)
                for n in local.nodes:           # seal the subtree except at its entry
                    if n == entry:
                        continue
                    for nb in work[n]:
                        work[n][nb]["weight"] = g[n][nb]["length"] * AVOID_FACTOR
                for n in inside:
                    if rep[n] in comp:
                        rep[n] = entry
    terms = {head, *rep.values()}
    tree = steiner_tree(work, list(terms), weight="weight", method="mehlhorn")
    union = nx.Graph()
    union.add_edges_from((a, b, {"length": g[a][b]["length"]}) for a, b in list(tree.edges) + local_edges)
    union = union.subgraph(nx.node_connected_component(union, head)).copy()
    tree = nx.minimum_spanning_tree(union, weight="length")
    keep = {head, *tx_nodes.values()}
    leaves = [n for n in tree if tree.degree(n) == 1 and n not in keep]
    while leaves:                                  # prune dangling non-terminal branches
        tree.remove_nodes_from(leaves)
        leaves = [n for n in tree if tree.degree(n) == 1 and n not in keep]
    return tree


def orient(tree: nx.Graph, head) -> dict:
    return {child: parent for parent, child in nx.bfs_edges(tree, head)}


def downstream_sets(tree: nx.Graph, head, parent: dict, node_tx: dict[tuple, set]) -> dict:
    """Transformers at or below each node."""
    below = {n: set(node_tx.get(n, ())) for n in tree}
    for n in reversed(list(nx.bfs_tree(tree, head))):
        if n in parent:
            below[parent[n]] |= below[n]
    return below


def place_device(dev_tx: set[int], dev_bldg: set[int], tx_nodes: dict[int, tuple], parent: dict,
                 below: dict, tx_bldgs: dict[int, set[int]]):
    """Branch point whose downstream buildings best match the buildings inside the
    observed footprint (Jaccard over buildings, so it doesn't depend on how many
    transformers a placement method creates)."""
    cands = set()
    for t in dev_tx:
        n = tx_nodes[t]
        while n in parent:
            cands.add(n)
            n = parent[n]
    best, best_j = None, 0.0
    for n in cands:
        down = set().union(*(tx_bldgs[t] for t in below[n])) if below[n] else set()
        j = len(down & dev_bldg) / len(down | dev_bldg)
        if j > best_j:
            best, best_j = n, j
    return best, best_j


def device_members(devices: gpd.GeoDataFrame, bld: gpd.GeoDataFrame, txs: pd.DataFrame,
                   tx_bldgs: dict[int, set[int]]) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Buildings inside each footprint, and the transformers most of whose buildings are."""
    cent = bld.geometry.centroid
    dev_tx, dev_bldg = {}, {}
    for d in devices.itertuples():
        inside = set(np.flatnonzero(cent.within(d.geometry).to_numpy()))
        txm = {t for t, b in tx_bldgs.items() if b and len(b & inside) / len(b) >= 0.5}
        if len(txm) >= 2 and len(inside) >= 2:
            dev_tx[d.id], dev_bldg[d.id] = txm, inside
    return dev_tx, dev_bldg


def score(tree, head, tx_nodes, dev_tx, dev_bldg, tx_bldgs, ids) -> list[float]:
    parent = orient(tree, head)
    node_tx = defaultdict(set)
    for t, n in tx_nodes.items():
        node_tx[n].add(t)
    below = downstream_sets(tree, head, parent, node_tx)
    return [place_device(dev_tx[d], dev_bldg[d], tx_nodes, parent, below, tx_bldgs)[1] for d in ids]


# ------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuit", nargs="?", default="PON-C008")
    ap.add_argument("--kv", type=float, help="primary line-to-line kV (default: from the chosen OSM substation)")
    ap.add_argument("--substation", help="OSM substation name to feed from (default: nearest DTE one)")
    ap.add_argument("--placement", choices=["auto", "block", "street", "kmeans"], default="auto",
                    help="transformer placement: auto (rear-lot 'block' or curbside 'street', whichever the "
                         "outage-located transformers show), or force one; 'kmeans' is the plain baseline")
    args = ap.parse_args()
    os.chdir(ROOT)
    name = args.circuit
    out = OUT / name if args.placement == "auto" else OUT / f"{name}-{args.placement}"
    dss_dir = out / "opendss"
    dss_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---- inputs (cached by pilot_circuit.py)
    circ = circuit_features(name)
    root = circ[circ["is_feeder"] == True].iloc[0]  # noqa: E712
    bbox = tuple(float(v) for v in np.round(gpd.GeoSeries([root.geometry], crs="EPSG:4326").to_crs(METRIC)
                                            .buffer(1000).to_crs("EPSG:4326").total_bounds, 4))
    hull = gpd.GeoSeries([root.geometry], crs="EPSG:4326").to_crs(METRIC).iloc[0]
    fps = footprints_for({name: bbox})[name].to_crs(METRIC)
    bld_all = overture_buildings(bbox, name).to_crs(METRIC)
    roads, power = osm_frames(osm(bbox, name))
    roads, power = roads.to_crs(METRIC), power.to_crs(METRIC)

    # ---- 1. service points and loads
    bld = primary_buildings(bld_all)
    bld = bld[bld.geometry.centroid.within(hull)].reset_index(drop=True)
    floors = pd.to_numeric(bld["num_floors"], errors="coerce").fillna(1).clip(1, 10)
    tagged_com = bld["subtype"].isin(COMMERCIAL_TAGS) | bld["class"].isin(COMMERCIAL_TAGS)
    bld["kind"] = np.where(tagged_com | (bld["area_m2"] >= COMMERCIAL_M2), "com",
                           np.where(bld["area_m2"] >= MULTIFAMILY_M2, "mf", "res"))
    bld["units"] = np.where(bld["kind"] == "mf", np.maximum(2, (bld["area_m2"] * floors / M2_PER_UNIT).round()), 1)
    bld["commercial"] = bld["kind"] != "res"          # served by its own 3-phase transformer
    customers = int(root["customers"])
    est_customers = int(bld.loc[bld["kind"] != "com", "units"].sum() + (bld["kind"] == "com").sum())
    bldg_per_cust = int((bld["kind"] == "res").sum()) / customers
    county = COUNTY.get(name.split("-")[0], "G2601250")
    print(f"{name}: {len(bld)} buildings ({(bld['kind'] == 'mf').sum()} multifamily, "
          f"{(bld['kind'] == 'com').sum()} commercial) ~{est_customers} customers vs {customers} "
          f"DTE customers; county {county}", file=sys.stderr)
    ecache = CACHE / "eulp"
    res_p = eulp.sample_profiles("res", "MI", county, N_RES_PROFILES, ecache,
                                 types=["Single-Family Detached", "Single-Family Attached"])
    com_p = eulp.sample_profiles("com", "MI", county, N_COM_PROFILES, ecache)
    prof_dir = dss_dir / "profiles"
    for p in res_p + com_p:
        eulp.write_profile_csv(p, prof_dir)
    pick = [com_p[rng.integers(len(com_p))] if k == "com" else res_p[rng.integers(len(res_p))]
            for k in bld["kind"]]
    bld["profile"] = [p.name for p in pick]
    ref_m2 = np.array([p.floor_m2 if p.floor_m2 == p.floor_m2 and p.floor_m2 > 0 else 150 for p in pick])
    base_kw = np.array([p.peak_kw for p in pick])
    area = (bld["area_m2"] * floors).to_numpy()
    bld["peak_kw"] = np.select(
        [bld["kind"] == "res", bld["kind"] == "mf"],
        [base_kw * np.clip(area / ref_m2, 0.5, 2),     # house: scale by floor area
         base_kw * bld["units"] * 0.6],                # flats: ~60% of a house each
        base_kw * np.clip(area / ref_m2, 0.25, 4))     # commercial: scale by floor area
    shapes = {p.name: p.pu for p in res_p + com_p}
    feeder_kw = np.zeros(eulp.STEPS)
    for prof, kw in bld.groupby("profile")["peak_kw"].sum().items():
        feeder_kw += shapes[prof] * kw
    peak_step = int(feeder_kw.argmax())
    coincident_kw = float(feeder_kw[peak_step])
    print(f"coincident peak {coincident_kw:,.0f} kW at step {peak_step} "
          f"(non-coincident {bld['peak_kw'].sum():,.0f} kW)", file=sys.stderr)

    # ---- 2. transformers
    tx_obs = fps[(fps["level"] == 1) & fps.geometry.centroid.within(hull)]
    tx_obs = tx_obs[tx_obs.geometry.map(lambda g: len(g.exterior.coords) == 9)]  # octagons = located
    med = float(tx_obs["customers"].median()) if len(tx_obs) else 10.0
    fr = frontage_roads(roads)
    layout = observed_layout(tx_obs, bld, bldg_per_cust, fr)
    placement = args.placement
    if placement == "auto":
        placement = "block" if layout["setback_m"] > REAR_LOT_SETBACK_M else "street"
    print(f"observed transformers: reach {layout['reach_m']} m, {layout['setback_m']} m from the street "
          f"-> {placement} placement", file=sys.stderr)
    if placement == "block":
        txs, owner = assign_transformers_block(bld, tx_obs, bldg_per_cust, med, fr, layout["reach_m"])
    elif placement == "street":
        txs, owner = assign_transformers_street(bld, tx_obs, bldg_per_cust, med, fr, layout["reach_m"])
    else:
        txs, owner = assign_transformers(bld, tx_obs, bldg_per_cust, med)
    bld["tx"] = owner
    load_by_tx = bld.groupby("tx")["peak_kw"].sum()
    count_by_tx = bld.groupby("tx").size()
    txs["peak_kw"] = txs["tx"].map(load_by_tx).fillna(0)
    txs["n_bldg"] = txs["tx"].map(count_by_tx).fillna(0).astype(int)
    txs = txs[txs["n_bldg"] > 0].reset_index(drop=True)
    txs["kva"] = [std_size(kw * (DIVERSITY if n > 4 else 1) / 0.95, TX_3PH_KVA if ph == 3 else TX_1PH_KVA)
                  for kw, n, ph in zip(txs["peak_kw"], txs["n_bldg"], txs["phases"])]

    # ---- 3. substation, voltage, road graph
    g = road_graph(roads, hull.buffer(1500))
    snap = Snapper(g)
    subs = power[(power["power"] == "substation")].copy()
    subs["dist"] = subs.geometry.distance(hull)
    subs["kvs"] = [sorted(float(v) / 1000 for v in str(x or "").split(";") if v.strip().isdigit())
                   for x in subs["voltage"]]
    dte = subs[subs["operator"].fillna("").str.contains("DTE")]
    pool = dte if len(dte) else subs
    if args.substation:
        pool = subs[subs["name"].fillna("").str.contains(args.substation, case=False)]
    need_kva = coincident_kw / 0.95

    def feasible(kvs):  # lowest distribution voltage at this substation that can carry the load
        return next((v for v in kvs if 2 < v < 35 and CIRCUIT_KVA.get(v, 12000) >= need_kva), None)
    pool = pool.sort_values("dist")
    choice = next(((r, feasible(r["kvs"])) for _, r in pool.iterrows() if feasible(r["kvs"])), None)
    if args.kv:
        sub, kv = pool.iloc[0], args.kv
    elif choice:
        sub, kv = choice
    else:  # nothing feasible nearby: nearest substation at its highest distribution voltage
        sub = pool.iloc[0]
        kv = max((v for v in sub["kvs"] if 2 < v < 35), default=13.2)
    kv_ln = kv / math.sqrt(3)
    skipped = [f"{r['name']} ({r['voltage']})" for _, r in pool.iterrows()
               if r["dist"] < sub["dist"]]
    if skipped:
        print(f"  nearer substations too small at their distribution voltage for {need_kva:,.0f} kVA: "
              + ", ".join(skipped), file=sys.stderr)
    head, _ = snap(np.array([sub.geometry.centroid.x, sub.geometry.centroid.y]))
    print(f"substation {sub['name']!s} ({sub['voltage']}), primary {kv} kV", file=sys.stderr)

    tx_nodes = {int(i): snap(np.array([x, y]))[0] for i, x, y in zip(txs.index, txs["x"], txs["y"])}
    devices = fps[fps["level"].isin([2, 3]) & fps.geometry.centroid.within(hull)].copy()
    devices["area"] = devices.area
    devices = devices[devices["area"] < hull.area * 0.9]
    tx_bldgs = {i: set(np.flatnonzero((bld["tx"] == tid).to_numpy())) for i, tid in zip(txs.index, txs["tx"])}
    dev_tx, dev_bldg = device_members(devices, bld, txs, tx_bldgs)
    members = dev_tx
    devices = devices[devices["id"].isin(members)]

    # ---- 4. held-out evaluation: constrained vs. plain Steiner (SMART-DS-like)
    base_tree = build_tree(g, head, tx_nodes, dev_tx, None)
    base_scores = score(base_tree, head, tx_nodes, dev_tx, dev_bldg, tx_bldgs, list(dev_tx))
    ids = list(members)
    rng.shuffle(ids)
    folds = [ids[::2], ids[1::2]]
    held = []
    for k in (0, 1):
        train = devices[~devices["id"].isin(folds[k])]
        t = build_tree(g, head, tx_nodes, dev_tx, train)
        held += score(t, head, tx_nodes, dev_tx, dev_bldg, tx_bldgs, folds[k])
    tree = build_tree(g, head, tx_nodes, dev_tx, devices)
    fit_scores = score(tree, head, tx_nodes, dev_tx, dev_bldg, tx_bldgs, list(dev_tx))
    evaluation = {
        "devices_evaluated": len(members),
        "plain_steiner_mean_jaccard": round(float(np.mean(base_scores)), 3) if base_scores else None,
        "outage_constrained_heldout_mean_jaccard": round(float(np.mean(held)), 3) if held else None,
        "outage_constrained_fit_mean_jaccard": round(float(np.mean(fit_scores)), 3) if fit_scores else None,
        "plain_share_ge_0.8": round(float(np.mean(np.array(base_scores) >= 0.8)), 2) if base_scores else None,
        "heldout_share_ge_0.8": round(float(np.mean(np.array(held) >= 0.8)), 2) if held else None,
        "primary_km_plain": round(sum(d["length"] for *_, d in base_tree.edges(data=True)) / 1000, 1),
        "primary_km_constrained": round(sum(d["length"] for *_, d in tree.edges(data=True)) / 1000, 1),
    }
    print(json.dumps(evaluation, indent=2), file=sys.stderr)

    # ---- 5. orient, place devices, phase
    parent = orient(tree, head)
    node_tx = defaultdict(set)
    for t, n in tx_nodes.items():
        node_tx[n].add(t)
    below = downstream_sets(tree, head, parent, node_tx)
    kw_below = {n: float(txs.loc[list(s), "peak_kw"].sum()) if s else 0.0 for n, s in below.items()}
    three_below = {n: bool((txs.loc[list(s), "phases"] == 3).any()) if s else False for n, s in below.items()}
    dev_at = {}
    for d in devices.itertuples():
        node, j = place_device(dev_tx[d.id], dev_bldg[d.id], tx_nodes, parent, below, tx_bldgs)
        if node is not None and node in parent and node not in dev_at:
            dev_at[node] = {"id": d.id, "type": d.dev_type, "jaccard": round(j, 2),
                            "obs_customers": int(d.customers)}
    coincidence = coincident_kw / float(bld["peak_kw"].sum())  # feeder-level diversity
    ph3 = {n: kw_below[n] * coincidence >= THREE_PHASE_KW or three_below[n] for n in parent}
    phase_of = {}
    phase_load = {"1": 0.0, "2": 0.0, "3": 0.0}
    for n in nx.bfs_tree(tree, head):
        if n == head:
            continue
        if ph3[n]:
            phase_of[n] = "123"
        elif parent[n] == head or ph3.get(parent[n], True):
            p = min(phase_load, key=phase_load.get)  # new lateral: least-loaded phase
            phase_of[n] = p
            phase_load[p] += kw_below[n]
        else:
            phase_of[n] = phase_of[parent[n]]

    # collapse chains of degree-2 nodes into single lines (keep terminals/devices/branches)
    keep = {head, *tx_nodes.values(), *dev_at, *(p for n, p in parent.items() if n in dev_at)}
    children = defaultdict(list)
    for c, p in parent.items():
        children[p].append(c)
    names = {head: "sourcebus"}
    counter = [0]

    def node_name(n):
        if n not in names:
            counter[0] += 1
            names[n] = f"n{counter[0]}"
        return names[n]

    segments = []  # (from, to, length_m, phases, coords, device)
    stack = [head]
    while stack:
        start = stack.pop()
        for c in children[start]:
            coords, length, cur = [start, c], g[start][c]["length"], c
            dev = dev_at.get(c)
            while (cur not in keep and len(children[cur]) == 1 and phase_of[children[cur][0]] == phase_of[cur]
                   and children[cur][0] not in dev_at):
                nxt = children[cur][0]
                length += g[cur][nxt]["length"]
                coords.append(nxt)
                cur = nxt
            segments.append((start, cur, length, phase_of[c], coords, dev))
            stack.append(cur)

    # ---- 6. write OpenDSS (SMART-DS layout)
    circuit_id = bus(name)
    lc = (ROOT / "synth" / "smartds_linecodes.dss").read_text()
    (dss_dir / "LineCodes.dss").write_text(lc)
    lines, coords_out = [], {}

    def ph_nodes(ph: str) -> str:
        return "." + ".".join(ph)

    for a, b, length, ph, pts, dev in segments:
        na, nb = node_name(a), node_name(b)
        n_ph = len(ph)
        if dev:
            dev_amps = kw_below[b] * coincidence / 0.95 * SIZING_MARGIN / (math.sqrt(3) * kv if n_ph == 3 else kv_ln)
            code = conductor(dev_amps, SWITCH_3PH)[0] if n_ph == 3 else "fuse_1_4"
            kind = "recloser" if dev["type"] == "Device (II)" else "fuse"
            mid = f"{nb}_up"
            lines.append(f"! observed {dev['type']} footprint {dev['id']} ({dev['obs_customers']} customers, "
                         f"match {dev['jaccard']})")
            lines.append(f"New Line.{kind}_{nb} Units=km Length=0.001 bus1={na}{ph_nodes(ph)} "
                         f"bus2={mid}{ph_nodes(ph)} switch=y enabled=y phases={n_ph} Linecode={code}")
            coords_out[mid] = a
            na = mid
        amps = kw_below[b] * coincidence / 0.95 * SIZING_MARGIN / (math.sqrt(3) * kv if n_ph == 3 else kv_ln)
        code = conductor(amps, PRIMARY_3PH)[0] if n_ph == 3 else "1P_OH_AL_ACSR_#2_Sparrow_1"
        lines.append(f"New Line.l(r:{nb}-{na}) Units=km Length={length / 1000:.5f} bus1={na}{ph_nodes(ph)} "
                     f"bus2={nb}{ph_nodes(ph)} switch=n enabled=y phases={n_ph} Linecode={code}")
        coords_out[na], coords_out[nb] = a, b

    trs, loads_out, sec_lines = [], [], []
    taps = []
    for t in txs.itertuples():
        n = tx_nodes[t.Index]
        hv = node_name(n)
        ph = phase_of.get(n, "123") if n != head else "123"
        p = ph if len(ph) == 1 else str(1 + t.Index % 3)   # 1-phase transformers on a 3-phase line rotate
        tap_m = math.dist(n, (t.x, t.y))
        if tap_m > TAP_MIN_M:  # primary tap from the road line to a transformer set back from it
            tap_bus = f"t{t.Index}hv"
            coords_out[tap_bus] = (t.x, t.y)
            tap_ph = "123" if t.phases == 3 else p
            code = "3P_OH_AL_ACSR_4/0_Penguin_3" if t.phases == 3 else "1P_OH_AL_ACSR_#2_Sparrow_1"
            taps.append(f"New Line.l(r:{tap_bus}-{hv}) Units=km Length={tap_m / 1000:.5f} bus1={hv}{ph_nodes(tap_ph)} "
                        f"bus2={tap_bus}{ph_nodes(tap_ph)} switch=n enabled=y phases={len(tap_ph)} Linecode={code}")
            hv = tap_bus
        lv = f"t{t.Index}lv"
        coords_out[lv] = (t.x, t.y)
        if t.phases == 3:
            lv_kv = 0.208 if t.kva <= 150 else 0.48
            trs.append(f"New Transformer.tr(r:t{t.Index}-{lv}) phases=3 windings=2 %loadloss=0.8 %noloadloss=0.3 "
                       f"wdg=1 conn=delta bus={hv}.1.2.3 kV={kv} kva={t.kva} wdg=2 conn=wye bus={lv}.1.2.3 "
                       f"kV={lv_kv} kva={t.kva} XHL=5.0")
        else:
            trs.append(f"New Transformer.tr(r:t{t.Index}-{lv}) phases=1 windings=3 %loadloss=0.8 %noloadloss=0.47 "
                       f"wdg=1 conn=wye bus={hv}.{p} kV={kv_ln:.4f} kva={t.kva} "
                       f"wdg=2 conn=wye bus={lv}.1.0 kV=0.12 kva={t.kva} "
                       f"wdg=3 conn=wye bus={lv}.0.2 kV=0.12 kva={t.kva} XHL=2.4 XLT=2.4 XHT=1.6")
        for b in bld[bld["tx"] == t.tx].itertuples():
            svc = f"b{b.Index}"
            c = b.geometry.centroid
            coords_out[svc] = (c.x, c.y)
            length_km = max(0.005, math.dist((t.x, t.y), (c.x, c.y)) * 1.2 / 1000)
            pf_kvar = b.peak_kw * 0.33  # ~0.95 power factor
            if t.phases == 3:
                lv_kv = 0.208 if t.kva <= 150 else 0.48
                code, runs = conductor(b.peak_kw / 0.95 / (math.sqrt(3) * lv_kv), SERVICE_3PH)
                par = f" normamps={dict((c, r) for r, c in SERVICE_3PH)[code] * runs}" if runs > 1 else ""
                sec_lines.append(f"New Line.l(r:{svc}-{lv}) Units=km Length={length_km / runs:.5f} bus1={lv}.1.2.3 "
                                 f"bus2={svc}.1.2.3 switch=n enabled=y phases=3 Linecode={code}{par}"
                                 + (f" ! {runs} parallel runs as one equivalent line" if runs > 1 else ""))
                loads_out.append(f"New Load.load_{svc} conn=wye bus1={svc}.1.2.3 kV={lv_kv} Vminpu=0.8 Vmaxpu=1.2 "
                                 f"model=1 kW={b.peak_kw:.3f} kvar={pf_kvar:.3f} Phases=3 yearly={b.profile}")
            else:
                code, _ = conductor(b.peak_kw / 2 / 0.95 / 0.12, SERVICE_1PH)
                sec_lines.append(f"New Line.l(r:{svc}-{lv}) Units=km Length={length_km:.5f} bus1={lv}.1.2 "
                                 f"bus2={svc}.1.2 switch=n enabled=y phases=2 Linecode={code}")
                for k in (1, 2):
                    loads_out.append(f"New Load.load_{svc}_{k} conn=wye bus1={svc}.{k} kV=0.12 Vminpu=0.8 "
                                     f"Vmaxpu=1.2 model=1 kW={b.peak_kw / 2:.3f} kvar={pf_kvar / 2:.3f} "
                                     f"Phases=1 yearly={b.profile}")

    (dss_dir / "Lines.dss").write_text("\n".join(lines + taps + sec_lines) + "\n")
    (dss_dir / "Transformers.dss").write_text("\n".join(trs) + "\n")
    (dss_dir / "Loads.dss").write_text("\n".join(loads_out) + "\n")
    used = sorted(set(bld["profile"]))
    (dss_dir / "LoadShapes.dss").write_text("\n".join(
        f"New Loadshape.{p} npts=35040 interval=0.25 mult=(file=profiles/{p}.csv)" for p in used) + "\n")
    xy = np.array([coords_out[k] for k in coords_out])
    ll = lonlat(gpd.GeoSeries([Point(*p) for p in xy], crs=METRIC))
    (dss_dir / "Buscoords.dss").write_text("\n".join(f"{k} {lo:.7f} {la:.7f}" for k, (lo, la) in zip(coords_out, ll)) + "\n")
    (dss_dir / "Master.dss").write_text(f"""Clear

! Synthetic feeder for inferred DTE circuit {name}, SMART-DS-style, constrained by outage data.
! Generated by dte-outages/synth/build_feeder.py. Synthetic: not DTE's actual network.
New Circuit.feeder_{circuit_id} bus1=sourcebus pu=1.03 basekV={kv} R1=1e-05 X1=1e-05 R0=1e-05 X0=1e-05

Redirect LineCodes.dss
Redirect Lines.dss
Redirect Transformers.dss
Redirect LoadShapes.dss
Redirect Loads.dss

Set Voltagebases=[0.12, 0.208, 0.48, {kv_ln:.4f}, {kv}]
Calcvoltagebases
Buscoords Buscoords.dss

! Peak snapshot (loads at their kW). For a year of 15-minute steps, as in SMART-DS:
! Solve mode=yearly stepsize=15m number=35040
Solve
""")

    # ---- 7. service-drop realism and power flow check
    sites = fps[fps["level"] == 0].geometry.centroid
    service = service_stats(bld, txs, fr, sites[sites.within(hull)])
    print(json.dumps({"service": service}, indent=2), file=sys.stderr)
    pf = power_flow(dss_dir, {k.lower(): v for k, v in shapes.items()}, peak_step)
    summary = {
        "circuit": name, "dte_customers": customers, "buildings": len(bld),
        "buildings_by_kind": bld["kind"].value_counts().to_dict(),
        "estimated_customers": est_customers,
        "buildings_per_customer": round(bldg_per_cust, 2),
        "coincident_peak_kw": round(coincident_kw), "peak_step_15min": peak_step,
        "substation": {"name": sub["name"], "voltage": sub["voltage"],
                       "km_from_circuit": round(float(sub["dist"]) / 1000, 2)},
        "primary_kv": kv,
        "placement": placement, "observed_layout": layout,
        "service": service,
        "transformers": {"total": len(txs), "observed": int(txs["observed"].sum()),
                         "by_placement": txs["placement"].value_counts().to_dict(),
                         "three_phase": int((txs["phases"] == 3).sum()),
                         "kva_total": float(txs["kva"].sum())},
        "primary_segments": len(segments),
        "devices_placed": len(dev_at),
        "peak_kw_noncoincident": round(float(bld["peak_kw"].sum())),
        "profiles": {"res": len(res_p), "com": len(com_p), "county": county},
        "evaluation": evaluation,
        "power_flow": pf,
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "evaluation"}, indent=2, default=str))
    write_network_map(out / "network.html", name, segments, g, dev_at, txs, bld, sub, head, hull, tx_nodes)


def power_flow(dss_dir: Path, shapes: dict[str, np.ndarray], step: int) -> dict:
    """Solve at the feeder's coincident-peak 15-minute step: every load at its own
    profile's value for that step (the Master's snapshot puts all loads at peak)."""
    import opendssdirect as dss
    cwd = os.getcwd()
    try:
        dss.Text.Command(f'Redirect "{dss_dir / "Master.dss"}"')
    finally:
        os.chdir(cwd)
    i = dss.Loads.First()
    while i:
        m = shapes[dss.Loads.Yearly()][step]
        dss.Loads.kW(dss.Loads.kW() * m)
        dss.Loads.kvar(dss.Loads.kvar() * m)
        i = dss.Loads.Next()
    dss.Solution.Solve()
    pu = np.array(dss.Circuit.AllBusMagPu())
    pu = pu[pu > 0.1]
    total = dss.Circuit.TotalPower()
    losses = dss.Circuit.Losses()
    overloads = []
    dss.Circuit.SetActiveClass("Line")
    for name in dss.ActiveClass.AllNames():
        dss.Circuit.SetActiveElement(f"Line.{name}")
        norm = dss.CktElement.NormalAmps()
        amps = max(dss.CktElement.CurrentsMagAng()[::2][: dss.CktElement.NumPhases()] or [0])
        if norm and amps > norm:
            overloads.append(name)
    return {"converged": bool(dss.Solution.Converged()), "buses": dss.Circuit.NumBuses(),
            "load_kw": round(-total[0]), "losses_kw": round(losses[0] / 1000, 1),
            "v_min_pu": round(float(pu.min()), 3), "v_max_pu": round(float(pu.max()), 3),
            "share_buses_below_0.95": round(float((pu < 0.95).mean()), 3),
            "lines_over_normamps": len(overloads)}


def write_network_map(path, name, segments, g, dev_at, txs, bld, sub, head, hull, tx_nodes) -> None:
    ll = lambda pts: lonlat(gpd.GeoSeries([Point(p) for p in pts], crs=METRIC)).tolist()
    prim = pd.DataFrame([{"path": ll(pts), "phases": len(ph),
                          "tip": f"{'3-phase' if len(ph) == 3 else 'Phase ' + ph} primary, {length:.0f} m"}
                         for a, b, length, ph, pts, dev in segments])
    prim["color"] = [[16, 66, 129] if p == 3 else [85, 152, 231] for p in prim["phases"]]
    prim["width"] = [8 if p == 3 else 3 for p in prim["phases"]]  # metres
    txp = lonlat(gpd.GeoSeries([Point(x, y) for x, y in zip(txs["x"], txs["y"])], crs=METRIC))
    tdf = pd.DataFrame({"lon": txp[:, 0], "lat": txp[:, 1],
                        "color": [[235, 104, 52] if o else [237, 161, 0] for o in txs["observed"]],
                        "tip": [f"{'Observed' if o else 'Synthetic'} transformer {k} kVA, {n} buildings"
                                for o, k, n in zip(txs["observed"], txs["kva"], txs["n_bldg"])]})
    bc = bld.geometry.centroid
    tx_xy = dict(zip(txs["tx"], zip(txs["x"], txs["y"])))
    ends = [p for c, t in zip(bc, bld["tx"]) for p in ((c.x, c.y), tx_xy[t])]
    ends_ll = ll(ends)
    sec = pd.DataFrame({"path": [ends_ll[i:i + 2] for i in range(0, len(ends_ll), 2)], "tip": "Service drop"})
    tap_pts = [(tx_nodes[i], (x, y)) for i, x, y in zip(txs.index, txs["x"], txs["y"])
               if math.dist(tx_nodes[i], (x, y)) > TAP_MIN_M]
    taps = pd.DataFrame({"path": [ll([a, b]) for a, b in tap_pts], "tip": "Primary tap to transformer"})
    dv = pd.DataFrame([{"pos": ll([n])[0], "tip": f"{d['type'].replace('Device', 'Observed device')} "
                        f"({d['obs_customers']} customers), match {d['jaccard']}"} for n, d in dev_at.items()])
    s = ll([(sub.geometry.centroid.x, sub.geometry.centroid.y)])[0]
    layers = [
        pdk.Layer("PathLayer", sec, get_path="path", get_color=[160, 160, 160], width_min_pixels=1),
        pdk.Layer("PathLayer", taps, get_path="path", get_color=[85, 152, 231], get_width=2, width_min_pixels=1),
        pdk.Layer("PathLayer", prim, get_path="path", get_color="color", get_width="width",
                  width_min_pixels=2, pickable=True),
        pdk.Layer("ScatterplotLayer", tdf, get_position=["lon", "lat"], get_fill_color="color", get_radius=12,
                  radius_min_pixels=3, pickable=True),
        pdk.Layer("ScatterplotLayer", pd.DataFrame({"lon": [s[0]], "lat": [s[1]], "tip": [f"Substation {sub['name']}"]}),
                  get_position=["lon", "lat"], get_fill_color=[74, 58, 167], get_radius=40, radius_min_pixels=8,
                  pickable=True),
    ]
    if len(dv):
        layers.append(pdk.Layer("ScatterplotLayer", dv, get_position="pos", get_fill_color=[227, 73, 72],
                                get_line_color=[255, 255, 255], stroked=True, get_radius=20, radius_min_pixels=6,
                                pickable=True))
    c = lonlat(gpd.GeoSeries([hull.centroid], crs=METRIC))[0]
    deck = pdk.Deck(layers=layers, map_style=pdk.map_styles.CARTO_LIGHT, tooltip={"text": "{tip}"},
                    initial_view_state=pdk.ViewState(latitude=c[1], longitude=c[0], zoom=14))
    deck.to_html(str(path), open_browser=False, notebook_display=False)
    legend = ("<div style='position:fixed;top:12px;left:12px;z-index:10;background:#fff;padding:10px 12px;"
              "border-radius:8px;font:12px system-ui,sans-serif;box-shadow:0 1px 4px #0003'>"
              f"<b>{name}: synthetic feeder</b><div>━ 3-phase primary (dark) · ─ 1-phase lateral (light)</div>"
              "<div><span style='color:#eb6834'>●</span> observed transformer · "
              "<span style='color:#eda100'>●</span> synthetic transformer</div>"
              "<div><span style='color:#e34948'>●</span> fuse/recloser at an observed outage device</div>"
              "<div><span style='color:#4a3aa7'>●</span> substation · grey = service drops</div></div>")
    path.write_text(path.read_text().replace("<body>", "<body>" + legend, 1))


if __name__ == "__main__":
    main()
