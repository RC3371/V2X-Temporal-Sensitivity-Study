"""
Phase 2: Time Compensation Evaluation using MCTrack Velocity Estimates.

For each temporal gap (200, 300, 400, 500 ms):
  1. Take consecutive keyframe pairs in each scene (500ms apart)
  2. Match GT objects across frames by instance_token
  3. Match tracker boxes to GT boxes at the earlier frame
  4. Interpolate GT target positions for sub-500ms gaps
  5. Compute cost_raw (no compensation) and cost_comp (with velocity compensation)
  6. Calculate reduction% and correlation%

nuScenes keyframes are at 2Hz (500ms). For gaps < 500ms, GT target
positions are linearly interpolated between consecutive keyframes.

Usage:
  python phase2_time_compensation.py \
    --results results/nuscenes/20260611_161106/results_for_motion.json \
    --dataroot data/nuscenes \
    --version v1.0-trainval \
    --vel_key velocity_kalman_cv
"""

import argparse
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from nuscenes.nuscenes import NuScenes


def get_sample_pairs_in_scene(nusc, scene):
    """Get consecutive keyframe pairs (token_t, token_t500) for a scene."""
    pairs = []
    sample_token = scene['first_sample_token']
    prev_token = None
    while sample_token:
        if prev_token is not None:
            pairs.append((prev_token, sample_token))
        prev_token = sample_token
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next'] if sample['next'] != '' else None
    return pairs


def get_gt_boxes_by_instance(nusc, sample_token):
    """Get GT boxes indexed by instance_token."""
    sample = nusc.get('sample', sample_token)
    boxes = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        velocity = nusc.box_velocity(ann_token)
        boxes[ann['instance_token']] = {
            'translation': np.array(ann['translation'][:2]),  # x, y
            'velocity': velocity[:2] if not np.any(np.isnan(velocity)) else np.array([0.0, 0.0]),
            'category': ann['category_name'],
        }
    return boxes


def match_tracker_to_gt(pred_boxes, gt_boxes_dict, max_dist=2.0):
    """
    Match tracker boxes to GT boxes using Hungarian matching.
    Returns list of (pred_box, instance_token) pairs.
    """
    if len(pred_boxes) == 0 or len(gt_boxes_dict) == 0:
        return []

    gt_instances = list(gt_boxes_dict.keys())
    gt_positions = [gt_boxes_dict[inst]['translation'] for inst in gt_instances]

    # Build cost matrix
    cost = np.zeros((len(pred_boxes), len(gt_instances)))
    for i, pb in enumerate(pred_boxes):
        pred_pos = np.array(pb['translation'][:2])
        for j, gt_pos in enumerate(gt_positions):
            cost[i, j] = np.linalg.norm(pred_pos - gt_pos)

    row_idx, col_idx = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] <= max_dist:
            matches.append((pred_boxes[r], gt_instances[c]))

    return matches


def evaluate_gap(nusc, pred_by_sample, gap_ms, vel_key, max_dist=2.0):
    """
    Evaluate time compensation at a specific temporal gap.

    For each consecutive keyframe pair (t, t+500ms):
      - Find objects present in both frames (by instance_token)
      - Match tracker boxes at time t to GT at time t
      - For each matched object also present at t+500ms:
        - Interpolate GT target at t+gap_ms
        - cost_raw = ||tracker_pos_t - gt_target||
        - cost_comp = ||(tracker_pos_t + vel * gap_s) - gt_target||

    Returns dict with aggregated metrics.
    """
    gap_s = gap_ms / 1000.0
    interp_ratio = gap_ms / 500.0  # how far between t and t+500ms

    all_cost_raw = []
    all_cost_comp = []
    all_gt_displacement = []

    for scene in nusc.scene:
        pairs = get_sample_pairs_in_scene(nusc, scene)

        for token_t, token_t500 in pairs:
            # Skip if tracker doesn't have results for this frame
            if token_t not in pred_by_sample:
                continue

            # GT at both frames
            gt_t = get_gt_boxes_by_instance(nusc, token_t)
            gt_t500 = get_gt_boxes_by_instance(nusc, token_t500)

            # Find objects present in both GT frames
            common_instances = set(gt_t.keys()) & set(gt_t500.keys())
            if len(common_instances) == 0:
                continue

            # Match tracker boxes to GT at time t
            pred_boxes = pred_by_sample[token_t]
            matches = match_tracker_to_gt(pred_boxes, gt_t, max_dist=max_dist)

            for pred_box, instance_token in matches:
                # Only evaluate if this object exists at t+500ms too
                if instance_token not in common_instances:
                    continue

                # Tracker position and velocity at time t
                tracker_pos = np.array(pred_box['translation'][:2])
                tracker_vel = np.array(pred_box[vel_key])

                # GT positions at t and t+500ms
                gt_pos_t = gt_t[instance_token]['translation']
                gt_pos_t500 = gt_t500[instance_token]['translation']

                # Interpolate GT target position at t + gap_ms
                gt_target = gt_pos_t + (gt_pos_t500 - gt_pos_t) * interp_ratio

                # GT displacement (how far the object actually moved in gap_ms)
                gt_disp = np.linalg.norm(gt_target - gt_pos_t)

                # Cost raw: tracker position at t vs GT target (no compensation)
                cost_raw = np.linalg.norm(tracker_pos - gt_target)

                # Cost comp: compensated position vs GT target
                compensated_pos = tracker_pos + tracker_vel * gap_s
                cost_comp = np.linalg.norm(compensated_pos - gt_target)

                all_cost_raw.append(cost_raw)
                all_cost_comp.append(cost_comp)
                all_gt_displacement.append(gt_disp)

    all_cost_raw = np.array(all_cost_raw)
    all_cost_comp = np.array(all_cost_comp)
    all_gt_displacement = np.array(all_gt_displacement)

    # Filter to moving objects only (displacement > 0.1m)
    moving_mask = all_gt_displacement > 0.1
    raw_moving = all_cost_raw[moving_mask]
    comp_moving = all_cost_comp[moving_mask]
    disp_moving = all_gt_displacement[moving_mask]

    # Compute metrics — all objects
    mean_raw = float(all_cost_raw.mean())
    mean_comp = float(all_cost_comp.mean())
    reduction_pct = float((mean_raw - mean_comp) / mean_raw * 100) if mean_raw > 0 else 0.0

    # Correlation: how well does compensated position track GT target?
    # Using Pearson correlation on the displacement vectors
    corr_raw = 0.0
    corr_comp = 0.0
    if len(all_cost_raw) > 2:
        # Correlation between GT displacement and raw/comp residual
        corr_raw_r, _ = pearsonr(all_gt_displacement, all_cost_raw)
        corr_comp_r, _ = pearsonr(all_gt_displacement, all_cost_comp)
        corr_raw = float(corr_raw_r)
        corr_comp = float(corr_comp_r)

    # Moving objects metrics
    mean_raw_moving = float(raw_moving.mean()) if len(raw_moving) > 0 else 0.0
    mean_comp_moving = float(comp_moving.mean()) if len(comp_moving) > 0 else 0.0
    reduction_moving = float((mean_raw_moving - mean_comp_moving) / mean_raw_moving * 100) if mean_raw_moving > 0 else 0.0

    return {
        'gap_ms': gap_ms,
        'total_pairs': len(all_cost_raw),
        'moving_pairs': int(moving_mask.sum()),
        'cost_raw': mean_raw,
        'cost_comp': mean_comp,
        'reduction_pct': reduction_pct,
        'corr_raw': corr_raw,
        'corr_comp': corr_comp,
        'cost_raw_moving': mean_raw_moving,
        'cost_comp_moving': mean_comp_moving,
        'reduction_pct_moving': reduction_moving,
        'median_raw': float(np.median(all_cost_raw)),
        'median_comp': float(np.median(all_cost_comp)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Path to results_for_motion.json')
    parser.add_argument('--dataroot', default='data/nuscenes', help='nuScenes dataroot')
    parser.add_argument('--version', default='v1.0-trainval', help='nuScenes version')
    parser.add_argument('--vel_key', default='velocity_kalman_cv',
                        choices=['velocity_det', 'velocity_kalman_cv', 'velocity_diff', 'velocity_curve'],
                        help='Which velocity estimate to use')
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
    print(f"  {len(nusc.sample)} samples, {len(nusc.scene)} scenes")

    print(f"\nUsing velocity method: {args.vel_key}")
    print(f"Max matching distance: {args.max_dist}m")

    # Evaluate each temporal gap
    gaps = [200, 300, 400, 500]
    all_results = []

    for gap in gaps:
        print(f"\nEvaluating gap = {gap}ms...")
        result = evaluate_gap(nusc, pred_by_sample, gap, args.vel_key, args.max_dist)
        all_results.append(result)
        print(f"  Matched pairs: {result['total_pairs']} (moving: {result['moving_pairs']})")

    # Print results table — All Objects
    print("\n" + "=" * 90)
    print(f"PHASE 2 RESULTS: Time Compensation with {args.vel_key}")
    print("=" * 90)

    print(f"\n--- All Objects ---")
    print(f"{'Gap (ms)':<10} {'Cost Raw (m)':<14} {'Cost Comp (m)':<15} {'Reduc %':<10} {'Corr Raw':<10} {'Corr Comp':<10}")
    print("-" * 69)
    for r in all_results:
        print(f"{r['gap_ms']:<10} {r['cost_raw']:<14.4f} {r['cost_comp']:<15.4f} {r['reduction_pct']:<10.2f} {r['corr_raw']:<10.4f} {r['corr_comp']:<10.4f}")

    print(f"\n--- Moving Objects Only ---")
    print(f"{'Gap (ms)':<10} {'Cost Raw (m)':<14} {'Cost Comp (m)':<15} {'Reduc %':<10}")
    print("-" * 49)
    for r in all_results:
        print(f"{r['gap_ms']:<10} {r['cost_raw_moving']:<14.4f} {r['cost_comp_moving']:<15.4f} {r['reduction_pct_moving']:<10.2f}")

    # Save results
    output = {
        'velocity_method': args.vel_key,
        'max_matching_dist': args.max_dist,
        'gaps': all_results,
    }
    out_path = args.results.replace('results_for_motion.json', f'phase2_time_comp_{args.vel_key}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
