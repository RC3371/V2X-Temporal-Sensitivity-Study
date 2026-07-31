# V2X Temporal Sensitivity Study — nuScenes

Sensitivity analysis of **time-gap compensation** for cooperative perception,
conducted at the UCLA Mobility Lab.

**Question:** in a V2X system, two agents observe the same scene at different
timestamps. If we propagate a tracked box forward by the time gap using the
tracker's own velocity estimate, how much of the position error do we recover —
and where does that recovery break down?

Two phases:

- **Phase 1** — how accurate are the tracker's velocity estimates to begin with?
- **Phase 2** — how well does compensation using those velocities close the gap,
  measured four different ways across four velocity methods.

---

## Layout

```
.
├── phase1_velocity_eval.py          # Phase 1
├── phase2_time_compensation.py      # Phase 2 — overall cost
├── phase2_speed_grouped.py          # Phase 2 — by speed group (+ chart)
├── phase2_iou.py                    # Phase 2 — BEV / 3D IoU
├── phase2_lon_lat_deviation.py      # Phase 2 — longitudinal / lateral error
├── export_scene.py                  # LiDAR + boxes → JSON for the BEV viewer
│
├── results/                         # reproducible outputs of the scripts above
└── analysis/                        # derived outputs, no generating script
```

---

## The four velocity methods

`phase1_velocity_eval.py` evaluates all four in one pass; the Phase 2 scripts
take one at a time via `--vel_key`. This is the main experimental axis.

| `vel_key` | Source |
|---|---|
| `velocity_det` | Detector's own velocity output |
| `velocity_kalman_cv` | Kalman filter, constant-velocity model *(default)* |
| `velocity_diff` | Finite difference between consecutive track positions |
| `velocity_curve` | Curve fit over track history |

---

## Setup

```bash
conda create -n mctrack-env python=3.10.14
conda activate mctrack-env
pip install -r requirements.txt
```

**On macOS, a bare `python` often resolves to the system Python 2.7**, which
fails on the f-strings in these scripts with a `SyntaxError` pointing at a
closing parenthesis. Activate the environment first and confirm:

```bash
which python && python --version     # expect .../mctrack-env/bin/python, 3.10.14
```

Or call the interpreter directly, skipping activation:

```bash
mctrack-env/bin/python phase2_speed_grouped.py ...
```

`shapely` is needed only by `phase2_iou.py`; the other scripts run without it.

---

## Data

Two inputs are needed for every Phase 1/2 script:

1. **nuScenes trainval** at `--dataroot` (default `data/nuscenes`), from
   https://nuscenes.org
2. **A tracker results file** in nuScenes submission format, extended with the
   four velocity fields — `results_for_motion.json`

Each box in the tracker file carries `translation`, `size`, `rotation`, the four
`velocity_*` fields, and `tracking_id`.

**The tracker results files are not in this repo.** `results_for_motion.json` is
~190 MB and `results.json` ~117 MB, both past GitHub's 100 MB per-file limit.
Regenerate them by running MCTrack (https://github.com/megvii-research/MCTrack)
on nuScenes with the pre-computed CenterPoint detections linked from its README.
The reference run used here was produced 2026-06-11.

To compare trackers, run the same scripts against a different
`results_for_motion.json` and pass `--tracker_name`.

---

## Running

```bash
# Phase 1 — velocity accuracy (all four methods in one pass)
python phase1_velocity_eval.py \
  --results results_for_motion.json \
  --dataroot data/nuscenes --version v1.0-trainval

# Phase 2 — overall cost
python phase2_time_compensation.py \
  --results results_for_motion.json \
  --dataroot data/nuscenes --vel_key velocity_kalman_cv

# Phase 2 — by speed group (writes .json + .png)
python phase2_speed_grouped.py \
  --results results_for_motion.json \
  --dataroot data/nuscenes --vel_key velocity_kalman_cv

# Phase 2 — IoU
python phase2_iou.py \
  --results results_for_motion.json \
  --dataroot data/nuscenes --vel_key velocity_kalman_cv --tracker_name MCTrack

# Phase 2 — longitudinal / lateral decomposition
python phase2_lon_lat_deviation.py \
  --results results_for_motion.json \
  --dataroot data/nuscenes --vel_key velocity_kalman_cv --tracker_name MCTrack

# BEV scene export
python export_scene.py --index 0 --dataroot data/nuscenes/v1.0-trainval \
  --max-points 15000 --out scene_export.json
```

Outputs are written next to the input `--results` file. To reproduce the full
sweep, run the two `--vel_key`-aware Phase 2 scripts once per method:
`velocity_det`, `velocity_kalman_cv`, `velocity_diff`, `velocity_curve`.

**Gaps are hardcoded** to `[200, 300, 400, 500]` ms inside each script's
`main()`. There is no `--gaps` flag; edit the list to change the sweep.

---

## Method

Common structure across all Phase 2 scripts:

1. Walk each scene's consecutive keyframe pairs `(t, t+500ms)`. nuScenes
   keyframes are 2 Hz.
2. Keep GT objects present in **both** frames, matched by `instance_token`.
3. Hungarian-match tracker boxes at `t` to GT at `t`, gated at `--max_dist`
   (default 2.0 m).
4. For each gap, linearly interpolate the GT target position between the two
   keyframes: `gt_target = gt_t + (gt_t500 − gt_t) · (gap/500)`.
5. Compare:
   - `cost_raw  = ‖tracker_pos − gt_target‖`
   - `cost_comp = ‖(tracker_pos + v·Δt) − gt_target‖`
6. Report reduction `(raw − comp)/raw × 100`.

GT speed for grouping is displacement over the keyframe interval,
`‖gt_t500 − gt_t‖ / 0.5` — an average over 500 ms, not an instantaneous speed.

Speed groups differ by script: `phase2_speed_grouped.py` buckets in **km/h**
(0–30, 30–60, 60+); `phase2_iou.py` and `phase2_lon_lat_deviation.py` bucket in
**mph** (0–20, 20–40, 40–60).

---

## `results/`

Reproducible outputs — rerunning the scripts above regenerates all of these.

| File | Produced by |
|---|---|
| `phase1_velocity_eval.json` | `phase1_velocity_eval.py` |
| `phase2_time_comp_velocity_{det,kalman_cv,diff,curve}.json` | `phase2_time_compensation.py` |
| `phase2_speed_grouped_velocity_{det,kalman_cv,diff,curve}.json` + `.png` | `phase2_speed_grouped.py` |
| `phase2_iou_MCTrack.json` | `phase2_iou.py` |
| `phase2_lonlat_MCTrack.json` | `phase2_lon_lat_deviation.py` |

---

## `analysis/`

Derived outputs kept separately because **no script in this repo generates
them** — they cannot be regenerated from this codebase as it stands.

| File | Contents |
|---|---|
| `dotplot_lonlat_500ms.json` | (lon, lat) deviation pairs at the 500 ms gap, for MCTrack and FastPoly, grouped 0–20 / 20–40 / 40–60 mph (400 / 400 / 26 points each) |
| `class_sensitivity.json` | Per-class sensitivity results |
| `lonlat_by_class.json` | Longitudinal / lateral deviation broken out by object class |
| `turns.json` | Turning-scenario results |
| `diag.json` | Diagnostic output |

The FastPoly figures in `dotplot_lonlat_500ms.json` come from running the Phase 2
scripts against a FastPoly `results_for_motion.json`, which is likewise not in
this repo.

---

## Not in this repo

- **Tracker results files** — see [Data](#data)
- **The generating code for `analysis/`**
- **The BEV point-cloud viewer** that consumes `export_scene.py` output
- **The `.npy` late-fusion pipeline** — a separate, earlier study on different
  data with a different matcher. Its cost-reduction percentages are not
  comparable to the nuScenes numbers here.

---

## Citation

UCLA Mobility Lab — PI: Prof. Jiaqi Ma.
Tracker: MCTrack (https://github.com/megvii-research/MCTrack), CenterPoint detections.
