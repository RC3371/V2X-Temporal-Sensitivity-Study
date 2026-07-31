"""
Phase 2 Speed-Grouped Analysis: Break down time compensation by speed group.

Groups objects by GT speed (0-30, 30-60, 60+ km/h) and computes
cost_raw, cost_comp, and reduction% per group per temporal gap.

Also generates a bar chart matching the previous evaluation format.

Usage:
  python phase2_speed_grouped.py \
    --results results/nuscenes/20260611_161106/results_for_motion.json \
    --dataroot data/nuscenes \
    --vel_key velocity_kalman_cv
"""

import argparse
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_sample_pairs_in_scene(nusc, scene):
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
    sample = nusc.get('sample', sample_token)
    boxes = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        velocity = nusc.box_velocity(ann_token)
        boxes[ann['instance_token']] = {
            'translation': np.array(ann['translation'][:2]),
            'velocity': velocity[:2] if not np.any(np.isnan(velocity)) else np.array([0.0, 0.0]),
            'category': ann['category_name'],
        }
    return boxes


def match_tracker_to_gt(pred_boxes, gt_boxes_dict, max_dist=2.0):
    if len(pred_boxes) == 0 or len(gt_boxes_dict) == 0:
        return []
    gt_instances = list(gt_boxes_dict.keys())
    gt_positions = [gt_boxes_dict[inst]['translation'] for inst in gt_instances]
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


def classify_speed_group(speed_mps):
    """Classify speed (m/s) into km/h groups."""
    speed_kmh = speed_mps * 3.6
    if speed_kmh < 30:
        return '0-30'
    elif speed_kmh < 60:
        return '30-60'
    else:
        return '60+'


def evaluate_speed_grouped(nusc, pred_by_sample, gaps, vel_key, max_dist=2.0):
    """
    Evaluate time compensation grouped by GT speed.
    Returns nested dict: results[gap_ms][speed_group] = {cost_raw, cost_comp, ...}
    """
    speed_groups = ['0-30', '30-60', '60+']

    # Accumulators: [gap][speed_group] -> lists
    raw_costs = {g: {sg: [] for sg in speed_groups} for g in gaps}
    comp_costs = {g: {sg: [] for sg in speed_groups} for g in gaps}

    processed = 0
    for scene in nusc.scene:
        pairs = get_sample_pairs_in_scene(nusc, scene)
        for token_t, token_t500 in pairs:
            if token_t not in pred_by_sample:
                continue

            gt_t = get_gt_boxes_by_instance(nusc, token_t)
            gt_t500 = get_gt_boxes_by_instance(nusc, token_t500)
            common_instances = set(gt_t.keys()) & set(gt_t500.keys())
            if len(common_instances) == 0:
                continue

            pred_boxes = pred_by_sample[token_t]
            matches = match_tracker_to_gt(pred_boxes, gt_t, max_dist=max_dist)

            for pred_box, instance_token in matches:
                if instance_token not in common_instances:
                    continue

                tracker_pos = np.array(pred_box['translation'][:2])
                tracker_vel = np.array(pred_box[vel_key])
                gt_pos_t = gt_t[instance_token]['translation']
                gt_pos_t500 = gt_t500[instance_token]['translation']

                # GT speed (average over the 500ms interval)
                gt_speed = np.linalg.norm(gt_pos_t500 - gt_pos_t) / 0.5  # m/s
                speed_group = classify_speed_group(gt_speed)

                for gap_ms in gaps:
                    gap_s = gap_ms / 1000.0
                    interp_ratio = gap_ms / 500.0
                    gt_target = gt_pos_t + (gt_pos_t500 - gt_pos_t) * interp_ratio

                    cost_raw = np.linalg.norm(tracker_pos - gt_target)
                    compensated_pos = tracker_pos + tracker_vel * gap_s
                    cost_comp = np.linalg.norm(compensated_pos - gt_target)

                    raw_costs[gap_ms][speed_group].append(cost_raw)
                    comp_costs[gap_ms][speed_group].append(cost_comp)

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(nusc.scene)} scenes...")

    # Aggregate
    results = {}
    for gap_ms in gaps:
        results[gap_ms] = {}
        for sg in speed_groups:
            raw = np.array(raw_costs[gap_ms][sg])
            comp = np.array(comp_costs[gap_ms][sg])
            if len(raw) > 0:
                mean_raw = float(raw.mean())
                mean_comp = float(comp.mean())
                reduc = float((mean_raw - mean_comp) / mean_raw * 100) if mean_raw > 0 else 0.0
                results[gap_ms][sg] = {
                    'count': len(raw),
                    'cost_raw': mean_raw,
                    'cost_comp': mean_comp,
                    'reduction_pct': reduc,
                    'median_raw': float(np.median(raw)),
                    'median_comp': float(np.median(comp)),
                }
            else:
                results[gap_ms][sg] = {'count': 0, 'cost_raw': 0, 'cost_comp': 0, 'reduction_pct': 0}

    return results


def plot_results(results, gaps, vel_key, out_path):
    """Generate bar chart matching previous evaluation format."""
    speed_groups = ['0-30', '30-60', '60+']
    speed_labels = ['0-30\nkm/h', '30-60\nkm/h', '60+\nkm/h']

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle('Match cost by speed group: raw vs. motion-compensated', fontsize=14, fontweight='bold')

    bar_width = 0.35
    x = np.arange(len(speed_groups))

    for idx, gap_ms in enumerate(gaps):
        ax = axes[idx]
        raw_vals = [results[gap_ms][sg]['cost_raw'] for sg in speed_groups]
        comp_vals = [results[gap_ms][sg]['cost_comp'] for sg in speed_groups]

        bars1 = ax.bar(x - bar_width / 2, raw_vals, bar_width, label='No compensation', color='#1f77b4')
        bars2 = ax.bar(x + bar_width / 2, comp_vals, bar_width, label='With compensation', color='#ff7f0e')

        ax.set_title(f'{gap_ms} ms', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(speed_labels, fontsize=10)
        ax.set_ylim(0, None)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(fontsize=8)

        if idx == 0:
            ax.set_ylabel('Average center-distance cost (m)', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to {out_path}")


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

    print("Loading MCTrack results...")
    with open(args.results) as f:
        results = json.load(f)
    pred_by_sample = results['results']
    print(f"  {len(pred_by_sample)} samples loaded")

    print("Loading nuScenes database...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"  {len(nusc.scene)} scenes")

    gaps = [200, 300, 400, 500]
    print(f"\nEvaluating {args.vel_key} by speed group...")
    results = evaluate_speed_grouped(nusc, pred_by_sample, gaps, args.vel_key, args.max_dist)

    # Print tables
    speed_groups = ['0-30', '30-60', '60+']
    print("\n" + "=" * 90)
    print(f"PHASE 2 SPEED-GROUPED RESULTS: {args.vel_key}")
    print("=" * 90)

    for gap_ms in gaps:
        print(f"\n--- Gap = {gap_ms}ms ---")
        print(f"{'Speed Group':<14} {'Count':<8} {'Cost Raw (m)':<14} {'Cost Comp (m)':<15} {'Reduc %':<10}")
        print("-" * 61)
        for sg in speed_groups:
            r = results[gap_ms][sg]
            print(f"{sg + ' km/h':<14} {r['count']:<8} {r['cost_raw']:<14.4f} {r['cost_comp']:<15.4f} {r['reduction_pct']:<10.2f}")

    # Generate chart
    chart_path = args.results.replace('results_for_motion.json', f'phase2_speed_grouped_{args.vel_key}.png')
    plot_results(results, gaps, args.vel_key, chart_path)

    # Save JSON
    json_path = args.results.replace('results_for_motion.json', f'phase2_speed_grouped_{args.vel_key}.json')
    # Convert gap keys to strings for JSON
    json_results = {str(k): v for k, v in results.items()}
    with open(json_path, 'w') as f:
        json.dump({'velocity_method': args.vel_key, 'results': json_results}, f, indent=2)
    print(f"Results saved to {json_path}")


if __name__ == '__main__':
    main()
