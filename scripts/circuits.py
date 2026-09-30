#!/usr/bin/env python3
"""Infer DTE's radial distribution topology (partial circuits) from outage polygons.

Why this works
--------------
DTE's outage polygons are convex hulls of the customers behind the device that
opened, and each outage is labelled with that device's class (DEV_TYPE_NAME).
On a radial feeder a downstream device's customers are a subset of every
upstream device's customers, so its hull lies inside theirs. A year of outages
therefore reveals a nested set of device "footprints" that can be assembled
into trees: substation > circuit > Device (II) > Device (I) > transformer > site.

Evidence used, strongest first:
  1. Observed transitions: a job whose polygon grows to an enclosing one
     (DTE rolling the outage up to an upstream device) or shrinks inside
     (partial restoration) directly links the two footprints.
  2. Containment: a higher-class footprint that contains >= 95% of the
     footprint's area and has at least as many customers. The tightest such
     hull is chosen as the parent.

Limits: hulls of neighbouring feeders overlap, so containment alone can be
ambiguous (flagged as confidence=ambiguous); DTE never publishes device IDs,
and feeders are reconfigured over time. Output is an approximation of the
protection hierarchy and service areas, not line routes.

Usage:
    pip install shapely
    python3 scripts/circuits.py            # writes dte/circuits/*
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from shapely.geometry import Polygon
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from history import SNAPSHOT, iter_snapshots  # noqa: E402

OUT_DIR = Path("dte/circuits")
LEVELS = {
    "Single Customer": 0, "Service Secondary": 1, "Service Transformer": 1,
    "Device (I)": 2, "Device (II)": 3, "Circuit Level": 4,
    "Subtransmission": 5, "Substation": 6, "Source": 7,
}
LEVEL_NAMES = {0: "Customer site", 1: "Transformer", 2: "Device (I)", 3: "Device (II)",
               4: "Circuit", 5: "Subtransmission", 6: "Substation", 7: "Source"}
CIRCUIT = 4
CONTAIN = 0.95      # share of a child's area that must lie inside its parent
MERGE_IOU = 0.9     # same-class footprints this similar are one device
MAP_MIN_LEVEL = 2   # footprints at or above this level get geometry in the GeoJSON

KX = 111.32 * math.cos(math.radians(42.5))  # km per degree near Detroit
KY = 110.57


def to_km(ring):
    return Polygon([(x * KX, y * KY) for x, y in ring]).buffer(0)


# ------------------------------------------------------------------ collect

def collect(snapshot_path: Path):
    """Replay every snapshot into footprints, transitions and job assignments."""
    fps: dict[str, dict] = {}
    transitions: Counter = Counter()  # (from_fp, to_fp) within one job
    job_last: dict[str, str] = {}
    job_best: dict[str, tuple[int, str]] = {}  # job -> (level, fp) of its widest extent

    for seen_at, snap in iter_snapshots(snapshot_path):
        for f in snap.get("features", []):
            p = f.get("properties") or {}
            g = f.get("geometry") or {}
            level = LEVELS.get(p.get("DEV_TYPE_NAME"))
            if p.get("NUM_CUST") is None or level is None or g.get("type") != "Polygon":
                continue
            ring = [[round(x, 5), round(y, 5)] for x, y in g["coordinates"][0]]
            key = hashlib.md5(json.dumps(ring).encode()).hexdigest()[:12]
            fp = fps.get(key)
            if fp is None:
                fp = fps[key] = {"ring": ring, "levels": Counter(), "customers": 0, "jobs": set(),
                                 "sc": Counter(), "first_seen": seen_at, "last_seen": seen_at}
            job = p["JOB_ID"]
            if job not in fp["jobs"]:
                fp["jobs"].add(job)
                fp["levels"][level] += 1
                if p.get("SERVICE_CENTER"):
                    fp["sc"][p["SERVICE_CENTER"]] += 1
            fp["customers"] = max(fp["customers"], int(p["NUM_CUST"]))
            fp["last_seen"] = seen_at

            prev = job_last.get(job)
            if prev and prev != key:
                transitions[(prev, key)] += 1
            job_last[job] = key
            if job not in job_best or level >= job_best[job][0]:
                job_best[job] = (level, key)
    return fps, transitions, {j: fp for j, (_, fp) in job_best.items()}


# -------------------------------------------------------------------- nodes

def build_nodes(fps: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str]]:
    """Merge near-identical same-class footprints into device nodes."""
    keys = list(fps)
    polys = [to_km(fps[k]["ring"]) for k in keys]
    level = [fps[k]["levels"].most_common(1)[0][0] for k in keys]
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = STRtree(polys)
    for i, poly in enumerate(polys):
        if poly.area == 0:
            continue
        for j in tree.query(poly):
            if j <= i or level[j] != level[i] or polys[j].area == 0:
                continue
            inter = poly.intersection(polys[j]).area
            if inter / (poly.area + polys[j].area - inter) >= MERGE_IOU:
                parent[find(j)] = find(i)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(keys)):
        groups[find(i)].append(i)

    nodes, alias = {}, {}
    for members in groups.values():
        rep = max(members, key=lambda i: len(fps[keys[i]]["jobs"]))
        rid = keys[rep]
        jobs = set().union(*(fps[keys[i]]["jobs"] for i in members))
        sc = sum((fps[keys[i]]["sc"] for i in members), Counter())
        nodes[rid] = {
            "id": rid, "ring": fps[rid]["ring"], "poly": polys[rep], "level": level[rep],
            "customers": max(fps[keys[i]]["customers"] for i in members),
            "events": len(jobs),
            "service_center": sc.most_common(1)[0][0] if sc else "",
            "first_seen": min(fps[keys[i]]["first_seen"] for i in members),
            "last_seen": max(fps[keys[i]]["last_seen"] for i in members),
        }
        for i in members:
            alias[keys[i]] = rid
    return nodes, alias


def evidence_edges(transitions: Counter, alias: dict[str, str], nodes: dict[str, dict]) -> Counter:
    """Directed (parent, child) counts from polygons that grew or shrank mid-outage."""
    edges: Counter = Counter()
    for (a, b), n in transitions.items():
        a, b = alias[a], alias[b]
        if a == b:
            continue
        pa, pb = nodes[a]["poly"], nodes[b]["poly"]
        if pa.area == 0 or pb.area == 0:
            continue
        if pa.intersection(pb).area / pa.area >= CONTAIN and pb.area > pa.area:
            edges[(b, a)] += n  # grew: b encloses a
        elif pb.intersection(pa).area / pb.area >= CONTAIN and pa.area > pb.area:
            edges[(a, b)] += n  # shrank: a encloses b
    return edges


# ------------------------------------------------------------------ parents

def assign_parents(nodes: dict[str, dict], edges: Counter, use_evidence: bool = True):
    ids = list(nodes)
    polys = [nodes[i]["poly"] for i in ids]
    tree = STRtree(polys)
    ev_by_child: dict[str, Counter] = defaultdict(Counter)
    if use_evidence:
        for (p, c), n in edges.items():
            ev_by_child[c][p] += n

    parents: dict[str, tuple[str | None, str]] = {}
    for idx, cid in enumerate(ids):
        child = nodes[cid]
        cpoly = polys[idx]
        if cpoly.area == 0:
            parents[cid] = (None, "none")
            continue
        cands = []
        for j in tree.query(cpoly):
            pid = ids[j]
            p = nodes[pid]
            if pid == cid or p["poly"].area <= cpoly.area:
                continue
            same_level_ok = p["level"] == child["level"] and ev_by_child[cid][pid] > 0
            if p["level"] < child["level"] or (p["level"] == child["level"] and not same_level_ok):
                continue
            if p["customers"] < 0.9 * child["customers"]:
                continue
            if p["poly"].intersection(cpoly).area / cpoly.area < CONTAIN:
                continue
            cands.append(pid)
        if not cands:
            parents[cid] = (None, "none")
            continue
        observed = [p for p in cands if ev_by_child[cid][p] > 0]
        if observed:
            best = max(observed, key=lambda p: (ev_by_child[cid][p], -nodes[p]["poly"].area))
            parents[cid] = (best, "observed")
            continue
        best = min(cands, key=lambda p: nodes[p]["poly"].area)
        bpoly = nodes[best]["poly"]
        # Another containing footprint that is not an ancestor of `best` means two
        # overlapping branches (e.g. neighbouring feeders) both claim this device.
        rival = any(nodes[p]["poly"].intersection(bpoly).area / bpoly.area < CONTAIN
                    for p in cands if p != best)
        parents[cid] = (best, "ambiguous" if rival else "inferred")
    return parents


def ancestors(nid: str, parents: dict) -> list[str]:
    chain = [nid]
    while (p := parents[chain[-1]][0]) is not None:
        chain.append(p)
    return chain


def validate(nodes, edges) -> dict:
    """Hold out the observed transitions: infer parents from containment alone and
    check how often the observed parent is recovered."""
    blind = assign_parents(nodes, edges, use_evidence=False)
    exact = chain = total = 0
    for (p, c), _ in edges.items():
        if nodes[p]["level"] <= nodes[c]["level"]:
            continue
        total += 1
        got = blind[c][0]
        exact += got == p
        chain += p in ancestors(c, blind)
    return {"held_out_edges": total,
            "exact_parent": round(exact / total, 3) if total else None,
            "parent_in_ancestor_chain": round(chain / total, 3) if total else None}


# ------------------------------------------------------------------ feeders

def feeders(nodes, parents) -> dict[str, dict]:
    """Assign every node below circuit level to a feeder: its circuit-level ancestor,
    or, when none was ever observed, the top of its fragment (a partial circuit)."""
    out = {}
    for nid, node in nodes.items():
        if node["level"] > CIRCUIT:
            out[nid] = {"feeder": None, "depth": 0}
            continue
        chain = ancestors(nid, parents)
        root = next((a for a in chain if nodes[a]["level"] == CIRCUIT), None)
        if root is None:
            root = [a for a in chain if nodes[a]["level"] < CIRCUIT][-1]
            if nodes[root]["level"] < 2:  # lone sites/transformers: not enough to call a circuit
                out[nid] = {"feeder": None, "depth": 0}
                continue
        out[nid] = {"feeder": root, "depth": chain.index(root)}
    return out


def name_feeders(nodes, fdr) -> dict[str, str]:
    roots = defaultdict(list)
    for nid, f in fdr.items():
        if f["feeder"]:
            roots[f["feeder"]].append(nid)
    by_sc = defaultdict(list)
    for root, members in roots.items():
        by_sc[nodes[root]["service_center"] or "UNK"].append((root, members))
    names = {}
    for sc, items in by_sc.items():
        for kind in ("C", "P"):
            group = [(r, m) for r, m in items if (nodes[r]["level"] == CIRCUIT) == (kind == "C")]
            group.sort(key=lambda rm: -sum(nodes[n]["events"] for n in rm[1]))
            for i, (root, _) in enumerate(group, 1):
                names[root] = f"{sc}-{kind}{i:03d}"
    return names


# ------------------------------------------------------------------- output

def write(nodes, parents, fdr, names, job_fp, alias, stats, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    members = defaultdict(Counter)
    events = Counter()
    for nid, f in fdr.items():
        if f["feeder"]:
            members[f["feeder"]][nodes[nid]["level"]] += 1
            events[f["feeder"]] += nodes[nid]["events"]

    def props(nid):
        n, (par, conf) = nodes[nid], parents[nid]
        f = fdr[nid]
        c = n["poly"].centroid
        return {
            "id": nid, "level": n["level"], "level_name": LEVEL_NAMES[n["level"]],
            "parent": par or "", "confidence": conf,
            "feeder": names.get(f["feeder"], "") if f["feeder"] else "",
            "depth": f["depth"], "customers": n["customers"], "events": n["events"],
            "service_center": n["service_center"], "first_seen": n["first_seen"], "last_seen": n["last_seen"],
            "lat": round(c.y / KY, 5), "lon": round(c.x / KX, 5), "area_km2": round(n["poly"].area, 4),
        }

    with (out_dir / "nodes.csv").open("w", newline="") as fh:
        rows = [props(n) for n in nodes]
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (-r["level"], r["feeder"], r["id"])))

    features = []
    for nid, n in nodes.items():
        if n["level"] < MAP_MIN_LEVEL:
            continue
        pr = props(nid)
        if fdr[nid]["feeder"] == nid:  # feeder roots carry the feeder's totals
            m = members[nid]
            pr.update(is_feeder=True, partial=n["level"] != CIRCUIT, feeder_events=events[nid],
                      devices=m[2] + m[3], transformers=m[1], sites=m[0])
        features.append({"type": "Feature", "properties": pr,
                         "geometry": {"type": "Polygon", "coordinates": [n["ring"]]}})
    (out_dir / "circuits.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                                          separators=(",", ":")))

    with (out_dir / "jobs.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["job_id", "footprint", "feeder"])
        for job, fp in sorted(job_fp.items()):
            nid = alias[fp]
            root = fdr[nid]["feeder"]
            w.writerow([job, nid, names.get(root, "") if root else ""])

    (out_dir / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    fps, transitions, job_fp = collect(args.snapshot)
    print(f"{len(fps)} distinct footprints, {len(transitions)} in-outage transitions", file=sys.stderr)
    nodes, alias = build_nodes(fps)
    edges = evidence_edges(transitions, alias, nodes)
    print(f"{len(nodes)} device nodes after merging, {len(edges)} observed parent/child links", file=sys.stderr)
    parents = assign_parents(nodes, edges)
    fdr = feeders(nodes, parents)
    names = name_feeders(nodes, fdr)

    conf = Counter(c for _, c in parents.values())
    roots = [r for r in names]
    stats = {
        "snapshots_from": min(n["first_seen"] for n in nodes.values()),
        "snapshots_to": max(n["last_seen"] for n in nodes.values()),
        "footprints": len(fps), "device_nodes": len(nodes),
        "nodes_by_level": {LEVEL_NAMES[k]: v for k, v in sorted(Counter(n["level"] for n in nodes.values()).items())},
        "parent_confidence": dict(conf),
        "circuits": sum(1 for r in roots if nodes[r]["level"] == CIRCUIT),
        "partial_circuits": sum(1 for r in roots if nodes[r]["level"] != CIRCUIT),
        "observed_links": len(edges),
        "assigned_to_circuit_by_level": {
            LEVEL_NAMES[lv]: round(sum(1 for n, f in fdr.items() if nodes[n]["level"] == lv and f["feeder"]
                                       and nodes[f["feeder"]]["level"] == CIRCUIT)
                                   / max(1, sum(1 for n in nodes.values() if n["level"] == lv)), 3)
            for lv in range(CIRCUIT)},
        "validation": validate(nodes, edges),
    }
    write(nodes, parents, fdr, names, job_fp, alias, stats, args.out)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
