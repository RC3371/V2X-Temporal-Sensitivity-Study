"""
Export one nuScenes sample's LiDAR point cloud + boxes to a compact JSON
that the BEV point-cloud viewer can ingest.

Usage:
    python export_scene.py                         # first sample, GT boxes
    python export_scene.py --index 42              # a different sample
    python export_scene.py --dataroot /path/to/nuscenes/v1.0-trainval
    python export_scene.py --max-points 8000       # smaller payload
    python export_scene.py --out my_scene.json

Requires: nuscenes-devkit, numpy  (pip install nuscenes-devkit)
"""

import argparse
import json

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="data/nuscenes/v1.0-trainval",
                    help="Path to the nuScenes data root (folder containing samples/, sweeps/, and the vX.Y json tables).")
    ap.add_argument("--version", default="v1.0-trainval",
                    help="nuScenes table version (e.g. v1.0-trainval, v1.0-mini).")
    ap.add_argument("--index", type=int, default=0,
                    help="Which sample (keyframe) to export, by position in nusc.sample.")
    ap.add_argument("--max-points", type=int, default=15000,
                    help="Cap on exported points (random subsample) to keep the JSON small.")
    ap.add_argument("--out", default="scene_export.json",
                    help="Output JSON path.")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    if not (0 <= args.index < len(nusc.sample)):
        raise SystemExit(
            "index {} out of range; this split has {} samples.".format(args.index, len(nusc.sample))
        )

    sample = nusc.sample[args.index]
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

    # --- point cloud (LiDAR sensor frame, x forward / y left / z up) ---
    pc = LidarPointCloud.from_file(args.dataroot + "/" + sd["filename"])
    xyz = pc.points[:3].T  # (N, 3)

    n_keep = min(args.max_points, len(xyz))
    if n_keep < len(xyz):
        idx = np.random.choice(len(xyz), size=n_keep, replace=False)
        xyz = xyz[idx]
    xyz = np.round(xyz, 2)

    # --- boxes (GT annotations, transformed into the same LiDAR frame) ---
    boxes = []
    for tok in sample["anns"]:
        b = nusc.get_box(tok)                       # global frame
        # bring the box into the LiDAR sensor frame so it lines up with the points
        b = _box_to_sensor_frame(nusc, b, sd)
        fwd = b.orientation.rotate([1.0, 0.0, 0.0])  # heading unit vector
        yaw = float(np.arctan2(fwd[1], fwd[0]))
        boxes.append({
            "id": tok[:4],
            "c": [round(float(b.center[0]), 2), round(float(b.center[1]), 2)],
            "z": round(float(b.center[2]), 2),
            "l": round(float(b.wlh[1]), 2),   # nuScenes wlh -> length is index 1
            "w": round(float(b.wlh[0]), 2),
            "h": round(float(b.wlh[2]), 2),
            "yaw": round(yaw, 3),
            "name": b.name,
        })

    out = {"points": xyz.tolist(), "boxes": boxes}
    with open(args.out, "w") as f:
        json.dump(out, f)

    print("wrote {}: {} points, {} boxes  (sample index {})".format(
        args.out, len(xyz), len(boxes), args.index))


def _box_to_sensor_frame(nusc, box, sd):
    """Transform a Box from the global frame into the LIDAR_TOP sensor frame."""
    from pyquaternion import Quaternion

    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

    # global -> ego
    box.translate(-np.array(ego["translation"]))
    box.rotate(Quaternion(ego["rotation"]).inverse)
    # ego -> sensor
    box.translate(-np.array(cal["translation"]))
    box.rotate(Quaternion(cal["rotation"]).inverse)
    return box


if __name__ == "__main__":
    main()
