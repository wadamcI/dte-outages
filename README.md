# dte-outages

An archive of DTE Energy's public power-outage map, collected for non-commercial
research at the University of Michigan. A GitHub Actions workflow polls the map's
public ArcGIS endpoint (GitHub runs the 15-minute schedule less often in practice)
and commits each change, so the git history is a record of every outage since
September 2025.

## Notice to DTE Energy

This project reads only the public data behind
[outagemap.dteenergy.com](https://outagemap.dteenergy.com), at a low request rate.
Every request identifies the project and the maintainer in its `User-Agent` and
`From` headers.

If you are a DTE Energy representative and would like this collection to stop,
to run at a different frequency, or to use a different endpoint, please contact
**wadamc@umich.edu**. I will act on the request promptly.

## Data

| Path | Contents |
| --- | --- |
| `dte/outages.geojson` | The latest snapshot: every active outage polygon with DTE's raw properties. |
| `dte/history/YYYY-MM.csv` | One row per outage job, grouped by the month it was first seen. Columns: `first_seen`/`last_seen` (UTC), `off_time`, `est_restore`, `customers_max`, `cause`, `service_center`, polygon centroid `lat`/`lon`, and others. |
| `dte/circuits/` | Inferred distribution circuits (see below). |

`first_seen`/`last_seen` are when the archiver observed the job, so durations are
accurate only to within the polling interval. Stale polygons that DTE serves
with no outage data are left out of the history.

### Inferred circuits

DTE doesn't publish its circuit topology, but the outage data implies part of it.
Each outage polygon is the convex hull of the customers behind the device that
opened, and each one is labelled with that device's class
(`DEV_TYPE_NAME`: Substation, Circuit Level, Device (II), Device (I),
Service Transformer, Single Customer). DTE's distribution is radial, so a
downstream device's customers are a subset of every upstream device's customers,
and its hull lies inside theirs. `scripts/circuits.py` replays the git history
and assembles those nested footprints into trees:

1. **Footprints.** Identical polygons recur across outages; near-identical ones
   of the same class (IoU ≥ 0.9) are merged into one device node.
2. **Observed links.** When an outage's polygon grows to an enclosing one
   (DTE rolling the job up to an upstream device) or shrinks inside (partial
   restoration), the two devices are linked directly.
3. **Containment.** Otherwise the parent is the tightest higher-class footprint
   that contains ≥ 95% of the device's area and serves at least as many customers.
4. **Circuits.** Every device is assigned to its Circuit Level ancestor, named like
   `CAN-C001` (service center, rank by outages). Device trees whose circuit hasn't
   had a full outage yet are *partial circuits* (`CAN-P001`).

| File | Contents |
| --- | --- |
| `circuits.geojson` | Footprints of Device (I) and above, with `parent`, `feeder`, `confidence`; circuit roots carry device and transformer counts |
| `nodes.csv` | Every device node, including transformers and customer sites (no geometry) |
| `jobs.csv` | Outage job → footprint → circuit |
| `summary.json` | Counts and validation results |

**How good is it?** For validation, the observed links from step 2 were held
out and parents were inferred from containment alone. On the 2025-09 to
2026-09 data that recovered the observed parent exactly **79%** of the time,
and the observed parent was somewhere in the inferred upstream chain 82% of
the time. `confidence` marks each link as `observed`, `inferred`, or
`ambiguous` (two overlapping branches, such as neighbouring circuits, both contain
the device). Caveats: convex hulls of neighbouring circuits overlap, DTE
reconfigures circuits over time, and device IDs are never published. Treat
the results as approximate service areas and a protection hierarchy, not as
line routes. A weekly workflow (`circuits.yml`) rebuilds them.

Raw files are available at
`https://raw.githubusercontent.com/wadamcI/dte-outages/main/dte/...`. The
[Intelligent Outage Dashboard](https://github.com/wadamcI/intelligent_outage_dashboard)
reads the data from there.

## Running locally

```bash
scripts/fetch_dte.sh                 # refresh dte/outages.geojson
python3 scripts/history.py update    # fold it into dte/history/
python3 scripts/history.py backfill  # rebuild dte/history/ from git (~30 s)
pip install -r requirements.txt && python3 scripts/circuits.py   # infer circuits (~90 s)
```

Requires `curl`, `jq` and Python 3.9+. Only `circuits.py` needs a package (shapely).

## Known issues

- **2026-09-23 to 2026-09-30:** no data. DTE's TLS certificate for
  `outagemap.serv.dteenergy.com` expired on 2026-09-23 and every fetch failed.
  The workflow now retries without certificate verification in that one case
  (`ALLOW_INSECURE_TLS_FALLBACK` in the workflow) and logs a warning. Turn the
  fallback off once DTE renews the certificate.
