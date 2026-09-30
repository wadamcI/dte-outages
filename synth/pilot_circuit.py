#!/usr/bin/env python3
"""Pilot: can open building/road data reproduce what DTE's outages say about a circuit?

For each inferred circuit given (default ANN-C001) this compares the outage footprints
with Overture Maps buildings and OpenStreetMap roads/power features:

  1. Customer sites  - single-customer outages are 62 m octagons around one point;
                       how close is that point to a building (vs. a shifted baseline)?
  2. Transformers    - are their footprints octagons around one point (a location)?
                       do building counts inside match the customers they serve?
  3. Devices         - customers vs. buildings inside each Device (I)/(II) footprint.
  4. Circuit         - buildings in the circuit footprint vs. its customers, and how
                       much of it is also claimed by neighbouring circuits.
  5. OSM power data  - substations, mapped lines and poles in the area.

Outputs synth/out/<CIRCUIT>/summary.json and map.html. Downloads are cached in
synth/cache/. Needs: geopandas shapely pyproj pydeck overturemaps requests.

    python3 synth/pilot_circuit.py RFD-C001 RFD-C020 PON-C008 ANN-C001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import requests
from shapely.geometry import LineString, Point, Polygon, box, shape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from history import SNAPSHOT, iter_snapshots  # noqa: E402

CACHE = ROOT / "synth" / "cache"
OUT = ROOT / "synth" / "out"
METRIC = "EPSG:32617"  # UTM 17N, metres
USER_AGENT = "dte-outages-pilot (+https://github.com/wadamcI/dte-outages; contact: wadamc@umich.edu)"
OVERPASS = ["https://overpass-api.de/api/interpreter",  # tried in order; the main one is often busy
            "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
LEVELS = {"Single Customer": 0, "Service Secondary": 1, "Service Transformer": 1,
          "Device (I)": 2, "Device (II)": 3, "Circuit Level": 4,
          "Subtransmission": 5, "Substation": 6, "Source": 7}
SITE_RADIUS_M = 62          # single-customer outages are octagons of this radius
NON_CUSTOMER_CLASSES = {"garage", "garages", "shed", "barn", "silo", "farm_auxiliary", "roof",
                        "carport", "greenhouse", "outbuilding", "storage_tank"}


# ------------------------------------------------------------------- inputs

def circuit_features(name: str) -> gpd.GeoDataFrame:
    g = json.loads((ROOT / "dte/circuits/circuits.geojson").read_text())
    feats = [f for f in g["features"] if f["properties"]["feeder"] == name]
    if not feats:
        sys.exit(f"circuit {name} not found in dte/circuits/circuits.geojson")
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


def all_circuit_roots() -> gpd.GeoDataFrame:
    g = json.loads((ROOT / "dte/circuits/circuits.geojson").read_text())
    feats = [f for f in g["features"] if f["properties"].get("is_feeder") and not f["properties"].get("partial")]
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


def footprints_for(bboxes: dict[str, tuple]) -> dict[str, gpd.GeoDataFrame]:
    """Every outage footprint (all device classes) inside each circuit's bbox, from
    the git history. Missing caches are filled with a single replay."""
    paths = {n: CACHE / f"footprints-{n}.geojson" for n in bboxes}
    todo = {n: box(*b) for n, b in bboxes.items() if not paths[n].exists()}
    if todo:
        found: dict[str, dict[str, dict]] = {n: {} for n in todo}
        for _, snap in iter_snapshots(SNAPSHOT):  # repo-relative; main() runs from ROOT
            for f in snap.get("features", []):
                p = f.get("properties") or {}
                g = f.get("geometry") or {}
                if p.get("NUM_CUST") is None or g.get("type") != "Polygon" or p.get("DEV_TYPE_NAME") not in LEVELS:
                    continue
                ring = [[round(x, 5), round(y, 5)] for x, y in g["coordinates"][0]]
                first = Point(*ring[0])
                for n, area in todo.items():
                    if not area.contains(first):
                        continue
                    key = hashlib.md5(json.dumps(ring).encode()).hexdigest()[:12]
                    fp = found[n].setdefault(key, {"ring": ring, "types": Counter(), "customers": 0, "jobs": set()})
                    if p["JOB_ID"] not in fp["jobs"]:
                        fp["jobs"].add(p["JOB_ID"])
                        fp["types"][p["DEV_TYPE_NAME"]] += 1
                    fp["customers"] = max(fp["customers"], int(p["NUM_CUST"]))
        CACHE.mkdir(parents=True, exist_ok=True)
        for n, fps in found.items():
            rows = [{"id": k, "dev_type": v["types"].most_common(1)[0][0], "customers": v["customers"],
                     "events": len(v["jobs"]), "geometry": Polygon(v["ring"])} for k, v in fps.items()]
            gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
            gdf["level"] = gdf["dev_type"].map(LEVELS)
            gdf.to_file(paths[n], driver="GeoJSON")
    return {n: gpd.read_file(paths[n]) for n in bboxes}


def overture_buildings(bbox, name: str) -> gpd.GeoDataFrame:
    path = CACHE / f"buildings-{name}.parquet"
    if not path.exists():
        from overturemaps import core
        reader = core.record_batch_reader("building", bbox)
        table = reader.read_all()
        gdf = gpd.GeoDataFrame.from_arrow(table)
        gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        gdf[["id", "subtype", "class", "num_floors", "height", "geometry"]].to_parquet(path)
    return gpd.read_parquet(path)


def osm(bbox, name: str) -> dict:
    path = CACHE / f"osm-{name}.json"
    if not path.exists():
        w, s, e, n = bbox
        q = f"""[out:json][timeout:180];
(
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|track)$"]({s},{w},{n},{e});
  nwr["power"]({s},{w},{n},{e});
);
out geom;"""
        errors = []
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=240)
                r.raise_for_status()
                r.json()
                break
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{url}: {exc}")
        else:
            sys.exit("Overpass query failed:\n" + "\n".join(errors))
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text)
    return json.loads(path.read_text())


def osm_frames(data: dict) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    roads, power = [], []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "node":
            geom = Point(el["lon"], el["lat"])
        elif el["type"] == "way" and el.get("geometry"):
            coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
            geom = Polygon(coords) if len(coords) > 3 and coords[0] == coords[-1] and "power" in tags \
                and tags["power"] in ("substation", "plant") else LineString(coords)
        elif el["type"] == "relation" and el.get("bounds"):
            b = el["bounds"]
            geom = box(b["minlon"], b["minlat"], b["maxlon"], b["maxlat"])
        else:
            continue
        row = {"osm_id": f"{el['type']}/{el['id']}", "geometry": geom, **{k: tags.get(k) for k in
               ("highway", "service", "power", "name", "operator", "voltage")}}
        (power if "power" in tags else roads).append(row)
    mk = lambda rows: gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else gpd.GeoDataFrame(
        {"geometry": []}, crs="EPSG:4326")
    return mk(roads), mk(power)


# ----------------------------------------------------------------- analysis

def primary_buildings(b: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Buildings likely to hold a meter: at least 50 m², not an outbuilding class, and
    not a detached garage/shed (a building under 90 m² with one at least twice its
    size within 20 m). Works at city density, where house lots are ~10 m apart."""
    b = b.copy()
    b["area_m2"] = b.area
    cand = b[(b["area_m2"] >= 50) & ~b["class"].isin(NON_CUSTOMER_CLASSES) &
             ~b["subtype"].isin(["agricultural", "outbuilding"])].copy()
    small = cand[cand["area_m2"] < 90]
    near = gpd.sjoin(small[["area_m2", "geometry"]], cand[["area_m2", "geometry"]].set_geometry(cand.buffer(20)),
                     predicate="intersects", how="inner", lsuffix="s", rsuffix="b")
    garages = near.index[near["area_m2_b"] >= 2 * near["area_m2_s"]].unique()
    return cand.drop(index=garages)


def nearest_dist(points: gpd.GeoSeries, targets: gpd.GeoDataFrame) -> np.ndarray:
    j = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=points.reset_index(drop=True), crs=points.crs),
                          targets[["geometry"]], how="left", distance_col="d")
    return j.groupby(level=0)["d"].min().to_numpy()


def is_octagon(geom) -> bool:
    ring = list(geom.exterior.coords)
    if len(ring) != 9:
        return False
    edges = np.hypot(*np.diff(np.array(ring), axis=0).T)
    return edges.std() / edges.mean() < 0.06


def summarize_ratio(customers: pd.Series, count: pd.Series) -> dict:
    ok = customers > 0
    ratio = (count[ok] / customers[ok])
    return {"n": int(ok.sum()), "median_buildings_per_customer": round(float(ratio.median()), 2),
            "within_x2": round(float(((ratio >= 0.5) & (ratio <= 2)).mean()), 2),
            # Spearman = Pearson on ranks (avoids a scipy dependency)
            "spearman": round(float(pd.Series(customers[ok].values).rank()
                                    .corr(pd.Series(count[ok].values).rank())), 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("circuits", nargs="*", default=["ANN-C001"])
    args = ap.parse_args()
    os.chdir(ROOT)

    setup = {}
    for name in args.circuits:
        circ = circuit_features(name)
        root = circ[circ["is_feeder"] == True].iloc[0]  # noqa: E712
        bbox = tuple(float(v) for v in np.round(gpd.GeoSeries([root.geometry], crs="EPSG:4326").to_crs(METRIC)
                                                .buffer(1000).to_crs("EPSG:4326").total_bounds, 4))
        setup[name] = (circ, root, bbox)
    footprints = footprints_for({n: v[2] for n, v in setup.items()})

    summaries = [run_circuit(n, *setup[n], footprints[n].to_crs(METRIC)) for n in args.circuits]
    if len(summaries) > 1:
        table = pd.DataFrame([compare_row(s) for s in summaries]).set_index("circuit")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "comparison.json").write_text(table.to_json(orient="index", indent=2) + "\n")
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(table.T.to_string())


def compare_row(s: dict) -> dict:
    return {
        "circuit": s["circuit"],
        "customers": s["customers"],
        "cust_per_km2": round(s["customers"] / max(s["area_km2"], 0.01)),
        "sites": s["sites"]["single_customer_sites_in_circuit"],
        "site_median_m_to_bldg": s["sites"]["median_m_to_building"],
        "site_within_10m": s["sites"]["share_within_10m"],
        "base30_within_10m": s["sites"]["baseline_shifted_30m_within_10m"],
        "site_inside_bldg": s["sites"]["share_inside_building"],
        "base30_inside_bldg": s["sites"]["baseline_shifted_30m_inside_building"],
        "site_median_m_to_road": s["sites"]["median_m_to_road"],
        "transformers": s["transformers"]["footprints_in_circuit"],
        "tx_octagon_share": s["transformers"]["share_regular_octagons"],
        "cust_per_tx": s["transformers"]["customers_median"],
        "tx_bldg_spearman": (s["transformers"]["customers_vs_primary_buildings"] or {}).get("spearman"),
        "tx_bldg_per_cust": (s["transformers"]["customers_vs_primary_buildings"] or {}).get("median_buildings_per_customer"),
        "tx_nearer_alley": s["transformers"]["share_nearer_alley_than_street"],
        "devices": s["devices"]["footprints_in_bbox"],
        "dev_bldg_per_cust": s["devices"]["raw"]["median_buildings_per_customer"],
        "dev_within_x2": s["devices"]["raw"]["within_x2"],
        "dev_spearman": s["devices"]["raw"]["spearman"],
        "circuit_bldg_per_cust": s["footprint"]["buildings_per_customer"],
        "large_bldg_share": s["footprint"]["share_large_buildings_400m2"],
        "shared_with_other_circuit": s["footprint"]["share_of_buildings_also_in_another_circuit"],
    }


def run_circuit(name, circ, root, bbox, fps) -> dict:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    hull_ll = root.geometry
    print(f"{name}: {root['customers']} customers, {root['area_km2']:.1f} km², bbox {bbox}", file=sys.stderr)
    bld = overture_buildings(bbox, name).to_crs(METRIC)
    roads, power = osm_frames(osm(bbox, name))
    roads, power = roads.to_crs(METRIC), power.to_crs(METRIC)
    hull = gpd.GeoSeries([hull_ll], crs="EPSG:4326").to_crs(METRIC).iloc[0]
    prim = primary_buildings(bld)
    print(f"{len(fps)} footprints, {len(bld)} buildings ({len(prim)} primary), "
          f"{len(roads)} roads, {len(power)} power features", file=sys.stderr)

    summary: dict = {"circuit": name, "customers": int(root["customers"]), "area_km2": round(root["area_km2"], 1),
                     "bbox": bbox, "buildings_in_bbox": len(bld), "primary_buildings_in_bbox": len(prim)}

    # 1. customer sites
    sites = fps[fps["level"] == 0].copy()
    sites["pt"] = sites.geometry.centroid
    sites_in = sites[sites["pt"].within(hull)]
    d_all = nearest_dist(sites_in["pt"], bld)
    d_prim = nearest_dist(sites_in["pt"], prim)
    rng = np.random.default_rng(0)
    ang = rng.uniform(0, 2 * np.pi, len(sites_in))

    def shifted(m):  # the same sites moved m metres in random directions
        return gpd.GeoSeries([Point(p.x + m * np.cos(a), p.y + m * np.sin(a))
                              for p, a in zip(sites_in["pt"], ang)], crs=METRIC)
    d_base = nearest_dist(shifted(150), bld)
    d_base30 = nearest_dist(shifted(30), bld)
    d_road = nearest_dist(sites_in["pt"], roads)
    inside = lambda pts: float(gpd.sjoin(gpd.GeoDataFrame(geometry=pts.reset_index(drop=True), crs=METRIC),
                                         bld[["geometry"]], predicate="within").index.nunique() / max(len(pts), 1))
    summary["sites"] = {
        "single_customer_sites_in_circuit": len(sites_in),
        "median_m_to_building": round(float(np.median(d_all)), 1),
        "share_within_10m": round(float((d_all <= 10).mean()), 2),
        "share_within_25m": round(float((d_all <= 25).mean()), 2),
        "median_m_to_road": round(float(np.median(d_road)), 1),
        "baseline_shifted_30m_within_10m": round(float((d_base30 <= 10).mean()), 2),
        "share_inside_building": round(inside(sites_in["pt"]), 2),
        "baseline_shifted_30m_inside_building": round(inside(shifted(30)), 2),
        "share_within_25m_of_primary": round(float((d_prim <= 25).mean()), 2),
        "baseline_shifted_150m_within_25m": round(float((d_base <= 25).mean()), 2),
        "baseline_median_m": round(float(np.median(d_base)), 1),
    }

    # 2. transformers
    tx = fps[(fps["level"] == 1) & fps.geometry.centroid.within(hull)].copy()
    tx["octagon"] = tx.geometry.map(is_octagon)
    tx["radius_m"] = tx.geometry.map(lambda g: float(np.median([Point(c).distance(g.centroid)
                                                                for c in g.exterior.coords[:-1]])))
    tx["buildings"] = tx.geometry.map(lambda g: int(prim.geometry.centroid.within(g).sum()))
    alleys = roads[roads["service"] == "alley"]
    streets = roads[roads["highway"] != "service"]
    nearer_alley = None
    if len(tx) and len(alleys):
        c = tx.geometry.centroid
        nearer_alley = round(float((nearest_dist(c, alleys) < nearest_dist(c, streets)).mean()), 2)
    summary["transformers"] = {
        "footprints_in_circuit": len(tx),
        "share_regular_octagons": round(float(tx["octagon"].mean()), 2) if len(tx) else None,
        "octagon_radius_m_median": round(float(tx.loc[tx["octagon"], "radius_m"].median()), 1) if tx["octagon"].any() else None,
        "customers_median": float(tx["customers"].median()) if len(tx) else None,
        "customers_vs_primary_buildings": summarize_ratio(tx["customers"], tx["buildings"]) if len(tx) else None,
        "osm_alleys_km": round(float(alleys.intersection(hull).length.sum() / 1000), 1),
        "share_nearer_alley_than_street": nearer_alley,
    }

    # 3. devices (any circuit, inside bbox) - raw footprint and shrunk by the site radius
    dev = fps[fps["level"].isin([2, 3])].copy()
    cent = prim.geometry.centroid
    dev["buildings"] = dev.geometry.map(lambda g: int(cent.within(g).sum()))
    dev["buildings_shrunk"] = dev.geometry.map(lambda g: int(cent.within(g.buffer(-SITE_RADIUS_M)).sum())
                                               if not g.buffer(-SITE_RADIUS_M).is_empty else 0)
    summary["devices"] = {
        "footprints_in_bbox": len(dev),
        "raw": summarize_ratio(dev["customers"], dev["buildings"]),
        "shrunk_62m": summarize_ratio(dev["customers"], dev["buildings_shrunk"]),
        "by_class": {t: summarize_ratio(g["customers"], g["buildings"]) for t, g in dev.groupby("dev_type")},
    }

    # 4. circuit footprint vs neighbours
    roots = all_circuit_roots().to_crs(METRIC)
    others = roots[(roots["feeder"] != name) & roots.intersects(hull)]
    in_hull = prim[cent.within(hull)]
    shared = in_hull.geometry.centroid.map(lambda p: bool(others.contains(p).any())) if len(others) else pd.Series(False, index=in_hull.index)
    served_pts = gpd.GeoSeries(sites_in["pt"].tolist() + tx.geometry.centroid.tolist(), crs=METRIC)
    summary["footprint"] = {
        "primary_buildings_in_footprint": len(in_hull),
        "buildings_per_customer": round(len(in_hull) / max(1, root["customers"]), 2),
        "share_large_buildings_400m2": round(float((in_hull["area_m2"] >= 400).mean()), 2) if len(in_hull) else None,
        "neighbouring_circuits_overlapping": len(others),
        "share_of_buildings_also_in_another_circuit": round(float(shared.mean()), 2) if len(in_hull) else None,
        "observed_service_points_convex_hull_km2": round(served_pts.union_all().convex_hull.area / 1e6, 1) if len(served_pts) > 2 else None,
    }

    # 5. OSM power
    pw_in = power[power.intersects(hull.buffer(2000))]
    lines = pw_in[pw_in["power"].isin(["line", "minor_line", "cable"])]
    summary["osm"] = {
        "substations_within_2km": pw_in.loc[pw_in["power"] == "substation", ["name", "operator", "voltage"]]
                                  .fillna("").to_dict("records"),
        "mapped_line_km_by_type": {k: round(v / 1000, 1) for k, v in lines.groupby("power").geometry
                                   .apply(lambda s: s.intersection(hull).length.sum()).items()},
        "poles_in_footprint": int(((pw_in["power"] == "pole") & pw_in.within(hull)).sum()),
        "transformers_mapped_in_footprint": int(((pw_in["power"] == "transformer") & pw_in.within(hull)).sum()),
        "road_km_in_footprint": round(float(roads.intersection(hull).length.sum() / 1000), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"  wrote {out / 'summary.json'}", file=sys.stderr)

    write_map(out / "map.html", hull, circ.to_crs(METRIC), fps, sites_in, tx, bld, prim, roads, power, name)
    return summary


def write_map(path, hull, circ, fps, sites, tx, bld, prim, roads, power, name) -> None:
    ll = lambda g: gpd.GeoSeries(g, crs=METRIC).to_crs("EPSG:4326")
    poly = lambda s: [[list(c) for c in geom.exterior.coords] for geom in s]
    layers = []

    b = ll(bld.geometry)
    bdf = pd.DataFrame({"polygon": poly(b), "primary": bld.index.isin(prim.index)})
    bdf["color"] = [[40, 40, 40, 200] if p else [150, 150, 150, 140] for p in bdf["primary"]]
    bdf["tip"] = np.where(bdf["primary"], "Building (likely customer)", "Outbuilding / small building")
    layers.append(pdk.Layer("PolygonLayer", bdf, get_polygon="polygon", get_fill_color="color",
                            stroked=False, pickable=True))

    r = ll(roads.geometry)
    rdf = pd.DataFrame({"path": [list(g.coords) for g in r], "tip": roads["highway"].fillna("").map("Road: {}".format)})
    layers.append(pdk.Layer("PathLayer", rdf, get_path="path", get_color=[200, 200, 190], width_min_pixels=1,
                            pickable=False))

    devs = circ[circ["level"].isin([2, 3])]
    ddf = pd.DataFrame({"polygon": poly(ll(devs.geometry)),
                        "tip": devs["level_name"] + ": " + devs["customers"].astype(str) + " customers"})
    layers.append(pdk.Layer("PolygonLayer", ddf, get_polygon="polygon", get_fill_color=[42, 120, 214, 30],
                            get_line_color=[42, 120, 214], line_width_min_pixels=1, pickable=True))
    layers.append(pdk.Layer("PolygonLayer", pd.DataFrame({"polygon": poly(ll([hull])), "tip": [f"{name} footprint"]}),
                            get_polygon="polygon", filled=False, get_line_color=[16, 66, 129], line_width_min_pixels=3))

    t = ll(tx.geometry.centroid)
    tdf = pd.DataFrame({"lon": t.x, "lat": t.y, "customers": tx["customers"].values,
                        "tip": "Transformer outage: " + tx["customers"].astype(str).values + " customers, "
                               + tx["buildings"].astype(str).values + " buildings inside"})
    layers.append(pdk.Layer("ScatterplotLayer", tdf, get_position=["lon", "lat"], get_fill_color=[235, 104, 52],
                            get_radius=25, radius_min_pixels=4, stroked=True, get_line_color=[255, 255, 255],
                            pickable=True))
    s = ll(sites["pt"])
    sdf = pd.DataFrame({"lon": s.x, "lat": s.y, "tip": "Customer site (single-customer outage)"})
    layers.append(pdk.Layer("ScatterplotLayer", sdf, get_position=["lon", "lat"], get_fill_color=[27, 175, 122],
                            get_radius=12, radius_min_pixels=3, pickable=True))

    pw = power.to_crs("EPSG:4326")
    lines = pw[pw.geometry.type == "LineString"]
    if len(lines):
        layers.append(pdk.Layer("PathLayer", pd.DataFrame({"path": [list(g.coords) for g in lines.geometry],
                                "tip": "OSM power " + lines["power"].fillna("")}),
                                get_path="path", get_color=[227, 73, 72], width_min_pixels=2, pickable=True))
    subs = pw[pw["power"] == "substation"]
    if len(subs):
        c = ll(power.loc[subs.index].geometry.centroid)  # centroid in metres, then to lon/lat
        layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame({"lon": c.x, "lat": c.y,
                                "tip": "Substation: " + subs["name"].fillna("(unnamed)")}),
                                get_position=["lon", "lat"], get_fill_color=[74, 58, 167], get_radius=60,
                                radius_min_pixels=7, pickable=True))

    center = ll([hull.centroid]).iloc[0]
    deck = pdk.Deck(layers=layers, map_style=pdk.map_styles.CARTO_LIGHT, tooltip={"text": "{tip}"},
                    initial_view_state=pdk.ViewState(latitude=center.y, longitude=center.x, zoom=11.3))
    deck.to_html(str(path), open_browser=False, notebook_display=False)
    swatch = lambda color, label, shape="50%": (
        f"<div><span style='display:inline-block;width:10px;height:10px;border-radius:{shape};"
        f"background:{color};margin-right:6px'></span>{label}</div>")
    legend = ("<div style='position:fixed;top:12px;left:12px;z-index:10;background:#fff;padding:10px 12px;"
              "border-radius:8px;font:12px system-ui,sans-serif;color:#222;box-shadow:0 1px 4px #0003'>"
              f"<b>{name} pilot</b>"
              + swatch("#10427f", "Inferred circuit footprint", "0")
              + swatch("rgba(42,120,214,.35)", "Device (I)/(II) footprints", "0")
              + swatch("#eb6834", "Transformer outage (octagon centre)")
              + swatch("#1baf7a", "Customer site (single-customer outage)")
              + swatch("#4a3aa7", "Substation (OSM)")
              + swatch("#e34948", "Power line (OSM, mostly 40–345 kV)", "0")
              + swatch("#282828", "Building, likely customer", "0")
              + swatch("#969696", "Outbuilding / small building", "0")
              + "</div>")
    html = path.read_text()
    path.write_text(html.replace("<body>", "<body>" + legend, 1))
    print(f"map written to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
