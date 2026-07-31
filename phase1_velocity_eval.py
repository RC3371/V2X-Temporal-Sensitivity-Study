"""
Phase 1: Compare MCTrack estimated velocities against nuScenes GT velocities.

For each sample frame:
  1. Load MCTrack tracked boxes (with multiple velocity estimates)
  2. Load GT annotated boxes (with GT velocity from devkit)
  3. Match by center point distance (Hungarian matching)
  4. Compute velocity error per matched pair
  5. Aggregate stats across all frames

Usage:
  python phase1_velocity_eval.py \
    --results results/nuscenes/20260611_161106/results_for_motion.json \
    --dataroot data/nuscenes \
    --version v1.0-trainval
"""

import argparse
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


def get_gt_boxes_and_velocity(nusc, sample_token):
    """Extract GT boxes with velocity for a given sample."""
    sample = nusc.get('sample', sample_token)
    gt_boxes = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        velocity = nusc.box_velocity(ann_token)  # (vx, vy, vz) global
        # Skip if velocity is NaN (first frame of a track, no prior frame)
        if np.any(np.isnan(velocity)):
            continue
        gt_boxes.append({
            'translation': ann['translation'],  # [x, y, z]
            'velocity': velocity[:2],            # [vx, vy] global
            'category': ann['category_name'],
            'instance_token': ann['instance_token'],
        })
    return gt_boxes


def match_boxes(pred_boxes, gt_boxes, max_dist=2.0):
    """
    Match predicted boxes to GT boxes using Hungarian matching
    on center point distance. Returns list of (pred_idx, gt_idx) pairs.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    # Build cost matrix: center point distance
    cost = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            pt = np.array(pb['translation'][:2])
            gt = np.array(gb['translation'][:2])
            cost[i, j] = np.linalg.norm(pt - gt)

    # Hungarian matching
    row_idx, col_idx = linear_sum_assignment(cost)

    # Filter by max distance threshold
    matches = []
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= max_dist:
            matches.append((r, c))

    return matches


def compute_velocity_errors(pred_boxes, gt_boxes, matches, vel_key):
    """
    For each matched pair, compute velocity error.
    Returns array of per-match errors.
    """
    errors = []
    for pi, gi in matches:
        v_pred = np.array(pred_boxes[pi][vel_key])
        v_gt = np.array(gt_boxes[gi]['velocity'])
        error = np.linalg.norm(v_pred - v_gt)
        errors.append(error)
    return np.array(errors)


def classify_motion(gt_boxes, matches, speed_thresh=0.5):
    """
    Classify matched GT boxes as static or moving based on GT speed.
    Returns dict with 'static' and 'moving' indices into the matches list.
    """
    groups = {'static': [], 'moving': []}
    for match_idx, (pi, gi) in enumerate(matches):
        speed = np.linalg.norm(gt_boxes[gi]['velocity'])
        if speed < speed_thresh:
            groups['static'].append(match_idx)
        else:
            groups['moving'].append(match_idx)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Path to results_for_motion.json')
    parser.add_argument('--dataroot', default='data/nuscenes', help='nuScenes dataroot')
    parser.add_argument('--version', default='v1.0-trainval', help='nuScenes version')
    parser.add_argument('--max_dist', type=float, default=2.0, help='Max matching distance (m)')
    args = parser.parse_args()

    # Load MCTrack results
    print("Loading MCTrack results...")
    with open(args.results) as f:
        results = json.load(f)
    pred_by_sample = results['results']
    print(f"  {len(pred_by_sample)} samples loaded")

    # Load nuScenes
    print("Loading nuScenes database...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"  {len(nusc.sample)} samples in database")

    # Velocity keys to evaluate
    vel_keys = ['velocity_det', 'velocity_kalman_cv', 'velocity_diff', 'velocity_curve']

    # Accumulators
    all_errors = {k: [] for k in vel_keys}
    all_errors_moving = {k: [] for k in vel_keys}
    all_errors_static = {k: [] for k in vel_keys}
    total_matches = 0
    total_moving = 0
    total_static = 0
    processed = 0

    print(f"\nEvaluating velocity across {len(pred_by_sample)} samples...")
    for sample_token, pred_boxes in pred_by_sample.items():
        # Get GT boxes with velocity
        gt_boxes = get_gt_boxes_and_velocity(nusc, sample_token)
        if len(gt_boxes) == 0:
            continue

        # Match predicted to GT
        matches = match_boxes(pred_boxes, gt_boxes, max_dist=args.max_dist)
        if len(matches) == 0:
            continue

        # Classify matches by motion
        motion_groups = classify_motion(gt_boxes, matches)
        total_matches += len(matches)
        total_moving += len(motion_groups['moving'])
        total_static += len(motion_groups['static'])

        # Compute errors per velocity method
        for vel_key in vel_keys:
            errors = compute_velocity_errors(pred_boxes, gt_boxes, matches, vel_key)
            all_errors[vel_key].extend(errors)

            # Moving objects only
            for idx in motion_groups['moving']:
                pi, gi = matches[idx]
                v_pred = np.array(pred_boxes[pi][vel_key])
                v_gt = np.array(gt_boxes[gi]['velocity'])
                all_errors_moving[vel_key].append(np.linalg.norm(v_pred - v_gt))

            # Static objects only
            for idx in motion_groups['static']:
                pi, gi = matches[idx]
                v_pred = np.array(pred_boxes[pi][vel_key])
                v_gt = np.array(gt_boxes[gi]['velocity'])
                all_errors_static[vel_key].append(np.linalg.norm(v_pred - v_gt))

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{len(pred_by_sample)} samples...")

    # Print results
    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS: MCTrack Velocity Accuracy vs GT")
    print("=" * 80)
    print(f"\nTotal matched boxes: {total_matches}")
    print(f"  Moving (speed >= 0.5 m/s): {total_moving}")
    print(f"  Static (speed <  0.5 m/s): {total_static}")

    print(f"\n{'Velocity Method':<22} {'Mean Err (m/s)':<16} {'Median':<10} {'Std':<10} {'P90':<10}")
    print("-" * 68)
    for vel_key in vel_keys:
        errs = np.array(all_errors[vel_key])
        print(f"{vel_key:<22} {errs.mean():<16.4f} {np.median(errs):<10.4f} {errs.std():<10.4f} {np.percentile(errs, 90):<10.4f}")

    print(f"\n--- Moving Objects Only ---")
    print(f"{'Velocity Method':<22} {'Mean Err (m/s)':<16} {'Median':<10} {'Std':<10} {'P90':<10}")
    print("-" * 68)
    for vel_key in vel_keys:
        errs = np.array(all_errors_moving[vel_key])
        if len(errs) > 0:
            print(f"{vel_key:<22} {errs.mean():<16.4f} {np.median(errs):<10.4f} {errs.std():<10.4f} {np.percentile(errs, 90):<10.4f}")

    print(f"\n--- Static Objects Only ---")
    print(f"{'Velocity Method':<22} {'Mean Err (m/s)':<16} {'Median':<10} {'Std':<10} {'P90':<10}")
    print("-" * 68)
    for vel_key in vel_keys:
        errs = np.array(all_errors_static[vel_key])
        if len(errs) > 0:
            print(f"{vel_key:<22} {errs.mean():<16.4f} {np.median(errs):<10.4f} {errs.std():<10.4f} {np.percentile(errs, 90):<10.4f}")

    # Save raw data for further analysis
    output = {
        'total_matches': total_matches,
        'total_moving': total_moving,
        'total_static': total_static,
        'velocity_methods': {}
    }
    for vel_key in vel_keys:
        errs = np.array(all_errors[vel_key])
        errs_m = np.array(all_errors_moving[vel_key])
        errs_s = np.array(all_errors_static[vel_key])
        output['velocity_methods'][vel_key] = {
            'all': {'mean': float(errs.mean()), 'median': float(np.median(errs)), 'std': float(errs.std()), 'p90': float(np.percentile(errs, 90))},
            'moving': {'mean': float(errs_m.mean()), 'median': float(np.median(errs_m)), 'std': float(errs_m.std()), 'p90': float(np.percentile(errs_m, 90))} if len(errs_m) > 0 else {},
            'static': {'mean': float(errs_s.mean()), 'median': float(np.median(errs_s)), 'std': float(errs_s.std()), 'p90': float(np.percentile(errs_s, 90))} if len(errs_s) > 0 else {},
        }

    out_path = args.results.replace('results_for_motion.json', 'phase1_velocity_eval.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
