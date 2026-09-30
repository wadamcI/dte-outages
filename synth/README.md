# Synthetic-network pilot

Can open data (Overture buildings, OpenStreetMap roads and power features, and
later ResStock/ComStock loads) reproduce what DTE's outages reveal about a
circuit, SMART-DS style but checked against observed outages?

```bash
pip install -r synth/requirements.txt
python3 synth/pilot_circuit.py RFD-C001 RFD-C020 PON-C008 ANN-C001
# -> synth/out/<CIRCUIT>/{summary.json,map.html} and synth/out/comparison.json
```

Downloads are cached in `synth/cache/`. Outputs and caches are git-ignored:
the building data is ODbL-licensed and large.

## Synthetic feeder builder (`build_feeder.py`)

```bash
python3 synth/pilot_circuit.py PON-C008     # once, to cache footprints/buildings/OSM
python3 synth/build_feeder.py PON-C008      # -> synth/out/PON-C008/opendss/, network.html
python3 synth/build_feeder.py PON-C008 --placement kmeans   # baseline, -> synth/out/PON-C008-kmeans/
```

This builds a SMART-DS-style feeder (same OpenDSS conventions, conductor library
and ResStock/ComStock loads) and constrains it with the outage data:

| Step | SMART-DS-style part | Outage-data improvement |
| --- | --- | --- |
| Loads | Overture buildings. Homes get a ResStock profile, multifamily ResStock × units, commercial ComStock; all sampled from the same county and scaled by floor area | Residential demand is calibrated to DTE's customer count |
| Transformers | Houses grouped around synthetic transformers | **Observed transformer locations** (octagon centres), each serving its **observed customer count**. The **layout** (rear-lot or curbside) and **reach** are measured from those transformers (see below) |
| Substation / voltage | Nearest DTE substation in OSM | Chosen at the **coincident peak**, so the voltage can carry the circuit |
| Primary | Minimal tree along OSM roads (approximate Steiner tree), with taps to transformers set back from the road | Each **observed device footprint** is wired first as a single-entry subtree |
| Protection | Not in SMART-DS | Fuses and reclosers at the branch point of each observed footprint (`switch=y` lines, as SMART-DS models fuses) |
| Phasing / sizing | 3-phase trunk, balanced 1-phase laterals, SMART-DS conductors sized by current | |

Output follows the SMART-DS feeder layout: `Master.dss`, `LineCodes.dss`,
`Lines.dss`, `Transformers.dss`, `LoadShapes.dss`, `Loads.dss`,
`Buscoords.dss` and `profiles/*.csv` (per-unit, 15-minute, 35,040 steps). Houses
are split-phase 120/240 V loads on center-tapped transformers, as in SMART-DS.

### Transformer placement (`--placement`)

The outage-located transformers show how this circuit's secondaries are laid out.
For each one, the reach is the distance to the farthest of its N nearest houses,
where N is its observed customer count; the setback is its distance from the
nearest street. In PON-C008 the 16 located transformers serve about 15.5
customers, reach 73 m (median) / 90 m (75th percentile), and sit **37 m behind
the street**: rear-lot easements, not curbside poles.

| Mode | How it places transformers |
| --- | --- |
| `auto` (default) | `block` if located transformers sit more than 20 m from the street, otherwise `street` |
| `block` | Rear-lot. Houses are grouped by city block (faces of the street network), so drops don't cross streets. Each block is clustered to the observed transformer size, and any group whose farthest house is beyond the measured reach is split. Tiny blocks share a neighbour's transformer if within reach |
| `street` | Curbside. Houses attach to their nearest street or alley and are cut into contiguous runs along it; the transformer sits on the street at the run's middle |
| `kmeans` | Baseline: k-means over all houses, transformer at the cluster centre |

PON-C008, all modes. "Sites" are the 84 single-customer outage points in the
circuit, which none of the placements use:

| | kmeans | street | **block (auto)** |
| --- | --- | --- | --- |
| Median service drop | 47 m | 39 m | **38 m** |
| Drops over 75 m | 11% | 11% | **4.5%** |
| Drops crossing a street | 35% | 12% | **9%** |
| Site to its transformer, median | 43 m | 44 m | **33 m** |
| Sites within 75 m of their transformer | 81% | 83% | **91%** |
| Houses per residential transformer | 15.4 | 9.0 | 10.0 (observed: ~13 buildings) |

Block placement is the most realistic on every secondary measure. Its
transformers are somewhat smaller than observed because this lakeside area has
many small blocks.

### PON-C008 result (block placement, 2026-09-30)

| | |
| --- | --- |
| Buildings | 2,273: 2,157 houses, 65 multifamily, 51 commercial; about 2,830 customers estimated vs. 2,519 DTE customers |
| Transformers | 332: 15 at observed locations, 201 rear-lot, 116 pad-mounts for commercial and multifamily buildings |
| Substation | Camden (40/13.2/4.8 kV), 13.2 kV. William Rensi (40/4.8 kV) is nearer, but 4.8 kV can't carry the 10.9 MVA peak as one circuit |
| Coincident peak | 10.4 MW of load, a January weekday at 18:00 (10.9 MW supplied at the substation, including losses) |
| Primary | 45 km including taps to rear-lot transformers; 43% single-phase |
| Peak power flow | converged; 0.927–1.03 pu; losses 4.3%; no line over its rating |
| One-day time series (peak day, 96 steps) | all converged; substation supply 6.8–10.9 MW; lowest voltage 0.911 pu |

### Does the outage device data help? Not measurably yet

Device footprints were split into two folds. Each fold was held out, the
network built with the other fold only, and each held-out device scored by the
best branch point: the Jaccard overlap between the **buildings** downstream of
it and the buildings inside the footprint.

| Placement | Plain Steiner tree | Outage-constrained, held-out devices |
| --- | --- | --- |
| kmeans | 0.41 | 0.43 |
| street | 0.37 | 0.33 |
| block | 0.38 | 0.34 |

With 11 devices in two folds this is within noise, so the device constraint
shows **no measurable benefit** on PON-C008. An earlier version scored
transformers instead of buildings and reported 0.40 → 0.50. That metric
depended on how many transformers a method creates and where their points fall
relative to a footprint's edge, so it is superseded. A real test needs many
circuits, or repeated random folds.

### Running OPF

OpenDSS solves power flow, not optimal power flow. For OPF, load the feeder
into [PowerModelsDistribution.jl](https://github.com/lanl-ansi/PowerModelsDistribution.jl),
which parses OpenDSS directly (untested here: Julia isn't installed):

```julia
using PowerModelsDistribution, Ipopt
eng = parse_file("synth/out/PON-C008/opendss/Master.dss")
result = solve_mc_opf(eng, ACPUPowerModel, Ipopt.Optimizer)
```

An OPF needs something to optimize, such as PV, batteries or reactive-power
control. Alternatively use `solve_mc_mld` (maximal load delivery), which
decides what can be restored after an outage. The observed devices are
switchable lines, so they can serve as switching points.

### Known gaps

- No voltage regulators or capacitor banks yet; SMART-DS feeders have them.
- Lateral phases are balanced by load; real phase assignments are unknown.
- The 4.8 kV vs. 13.2 kV choice is inferred from capacity. It isn't observed.
- Only one feeder per circuit. Detroit's large 4.8 kV "circuits" need
  splitting into several feeders.

## Dense vs. rural (2026-09-30)

The three densest well-observed circuits are compared with the rural ANN-C001.
Circuits are chosen from those with at least 5 devices, 15 transformers and
500 customers.

| | RFD-C001 (NW Detroit) | RFD-C020 (W Detroit) | PON-C008 (Pontiac) | ANN-C001 (rural) |
| --- | --- | --- | --- | --- |
| Customers / per km² | 6,263 / 870 | 1,115 / 1,239 | 2,519 / 741 | 1,497 / 6 |
| Footprint: likely-customer buildings per customer | **1.30** | **0.88** | **0.90** | 2.72 |
| Devices: rank correlation, buildings vs. customers | **0.82** | **0.86** | **0.83** | 0.73 |
| Devices: median buildings per customer | 0.70 | 0.54 | 0.44 | 1.10 |
| Customers per transformer (median) | 18.5 | 19 | 16.5 | 2 |
| Transformer outages drawn as 62 m octagons | 81% | 27% | 80% | 97% |
| Transformers nearer an alley than a street | 17% | **59%** | 0% | 0% |
| Buildings also inside another circuit's footprint | 17% | **94%** | 0% | 2% |
| Customer site inside a building (vs. same point shifted 30 m) | 8% (20%) | 8% (17%) | 14% (26%) | 1% (4%) |

**Dense areas work better for:**
- **Footprints.** They are tight: about 0.9–1.3 likely-customer buildings per
  customer, against 2.7 in the rural circuit.
- **Building counts.** They rank devices by size better (rank correlation
  0.82–0.86).
- **Transformers.** They serve realistic groups of 16–19 customers.
- **Line layout.** In RFD-C020, transformer outages sit in rows down the
  mid-block alleys, showing Detroit's rear-lot distribution directly.

**New problems in dense areas:**
- **Multi-customer buildings.** Devices hold only 0.44–0.70 buildings per
  customer, because of duplexes, flats and small apartment buildings. Building
  counts need a customers-per-building calibration, perhaps by building size
  or from Census housing units per block.
- **Customer-site points are not on buildings.** They land inside a building
  *less* often than randomly shifted points, and about 6 m from the nearest
  one. They look like meter or service-drop positions on the lot, so snapping
  to the nearest building is plausible but can't be checked from this data.
- **Overlapping circuit footprints.** 94% of RFD-C020's buildings are also
  inside another circuit's footprint. Assigning buildings to circuits needs
  the device and transformer structure, not just the circuit's convex hull.
- **Detroit's 4.8 kV system.** The nearby substations are 24/4.8 kV (Puritan,
  Appoline, Lauder, Woodside, Turner, Chicago Blvd, Decatur). Circuits there
  are small, so a "circuit" of 6,263 customers (RFD-C001) probably spans
  several real circuits.
- **OSM has almost no distribution lines** in these areas (under 3 km each).
  Lines must be routed along streets and alleys, and alleys are only partly
  mapped.

**Recommendation:** develop the synthesis on suburban circuits like PON-C008
first. They have tight footprints, clean transformer octagons, no overlap and
front-lot lines. PON-C008's voltage is unknown: nearby substations are Camden
(40/13.2/4.8 kV), Bartlett (40/8.32 kV) and William Rensi (40/4.8 kV). Handle
alley-fed Detroit 4.8 kV as a second case.

## ANN-C001 results (2026-09-30)

These numbers are from the first pilot run, which used a 10 m clustering rule
for buildings. The table above uses the later detached-garage rule, which
counts rural barns as buildings; that is why ANN-C001's ratios differ.

ANN-C001 is a rural circuit around Clinton, Macon and Saline: 1,497 customers,
234 km² footprint.

| Check | Result | Meaning |
| --- | --- | --- |
| Single-customer outage shape | Regular octagons, radius 62 m | Each center is a real service-point location. |
| Customer site → nearest building | median 32 m; 76% within 50 m (38% within 25 m vs. 10% for points shifted 150 m) | Sites are at real homes, but the point sits **between house and road** (median 21 m to a road), probably at the meter or service drop. Snap to building *and* road. |
| Transformer outage shape | **97%** regular octagons, radius 62.2 m | DTE draws transformer outages around the **transformer's location**, so we get 108 real transformer positions for this circuit. They line up along roads. |
| Customers per transformer | median 2 | Rural pole-top transformers. Buildings inside the 62 m octagon only weakly predict customers (Spearman 0.26), because the octagon marks the transformer and doesn't cover its customers. |
| Device footprints (58 in the area) | median 0.84 buildings per customer; 72% within a factor of 2; Spearman **0.73** (Device (I): 0.88) | Device hulls behave like customer hulls, and building counts track customer counts well. |
| Circuit footprint | 2,480 likely-customer buildings vs. 1,497 customers (1.66×); only 3% also inside a neighbouring circuit's footprint | The convex hull is loose in rural areas. Some buildings belong to the **Village of Clinton municipal utility** (two 40/4.16 kV substations in OSM), not DTE. |
| OSM substations | Macon (DTE, 40/13.2 kV) is the most central to the circuit's transformers; Ramsey (DTE, 40/13.2 kV) is also inside | Macon is the probable source. The footprint may contain parts of two circuits. |
| OSM lines | 86 km mapped, almost all 40–345 kV; 0.1 km at 13.2 kV | OSM has subtransmission but **not the distribution circuits**, so the circuit must be routed along roads. |

**Verdict:** worth continuing. The outage data provides exact customer-site and
transformer locations, which is more than SMART-DS started from. Next steps:

1. Remove non-DTE territory (municipal utilities such as the Village of Clinton)
   and use parcels to separate homes from outbuildings.
2. Assign buildings to transformers: each transformer serves its observed
   customer count, filled with the nearest unassigned buildings along the road.
3. Route primary lines along roads from the substation to the transformers
   (a Steiner tree on the road graph), with protective devices placed to match
   observed Device (I)/(II) footprints.
4. Attach ResStock/ComStock load profiles by building type and area, export to
   OpenDSS, and validate by holding out outages.
