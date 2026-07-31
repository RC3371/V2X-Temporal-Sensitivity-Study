"""
Phase 2 Extended: Longitudinal / Lateral Deviation Decomposition
=================================================================

For each matched (tracker, GT) pair after time compensation, decompose the
position error vector delta_p = p_compensated - p_gt into two components
relative to the GT heading (direction of travel):

  u_lon = (cos theta, sin theta)      longitudinal unit vector (along heading)
  u_lat = (-sin theta, cos theta)     lateral unit vector (perpendicular)

  d_lon = delta_p . u_lon = dx*cos + dy*sin
  d_lat = delta_p . u_lat = -dx*sin + dy*cos

Sanity: d_lon^2 + d_lat^2 == |delta_p|^2 (== center distance^2)

Metrics reported per (temporal gap x speed group):
  - MAX position deviation after compensation  (primary - worst-case safety)
  - MEAN center-distance cost                  (supplementary)
  - MAX / MEAN longitudinal deviation
  - MAX / MEAN lateral deviation

Speed groups (mph): 0-20, 20-40, 40-60
Temporal gaps (ms): 200, 300, 400, 500

Usage:
  python phase2_lon_lat_deviation.py \
    --results results/nuscenes/20260611_161106/results_for_motion.json \
    --dataroot data/nuscenes \
    --vel_key velocity_kalman_cv \
    --tracker_name MCTrack
"""

import argparse
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


MPH = 2.23694  # m/s -> mph


def quaternion_to_yaw(q):
    """Extract yaw (heading) from a nuScenes quaternion [w, x, y, z]."""
    return Quaternion(q).yaw_pitch_roll[0]


def speed_group_mph(speed_mps):
    mph = speed_mps * MPH
    if mph < 20:
        return '0-20'
    elif mph < 40:
        return '20-40'
    elif mph < 60:
        return '40-60'
    else:
        return '60+'


def get_sample_pairs_in_scene(nusc, scene):
    pairs = []
    tok = scene['first_sample_token']
    prev = None
    while tok:
        if prev:
            pairs.append((prev, tok))
        prev = tok
        s = nusc.get('sample', tok)
        tok = s['next'] if s['next'] != '' else None
    return pairs


def get_gt_by_instance(nusc, sample_token):
    """GT boxes indexed by instance_token: position, yaw."""
    sample = nusc.get('sample', sample_token)
    out = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        out[ann['instance_token']] = {
            'pos': np.array(ann['translation'][:2]),
            'yaw': quaternion_to_yaw(ann['rotation']),
        }
    return out


def decompose(delta, yaw):
    """Project error vector delta=(dx,dy) onto longitudinal/lateral axes."""
    c, s = np.cos(yaw), np.sin(yaw)
    d_lon = delta[0] * c + delta[1] * s
    d_lat = -delta[0] * s + delta[1] * c
    return d_lon, d_lat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--dataroot', default='data/nuscenes')
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument('--vel_key', default='velocity_kalman_cv')
    parser.add_argument('--tracker_name', default='tracker')
    parser.add_argument('--max_dist', type=float, default=2.0)
    args = parser.parse_args()

    print("Loading results...")
    with open(args.results) as f:
        pred_by_sample = json.load(f)['results']

    print("Loading nuScenes...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    gaps = [200, 300, 400, 500]
    groups = ['0-20', '20-40', '40-60']

    # accumulators: [gap][group] -> lists of signed deviations & center dist
    acc = {g: {sg: {'lon': [], 'lat': [], 'center': []} for sg in groups} for g in gaps}

    for scene in nusc.scene:
        for t0, t1 in get_sample_pairs_in_scene(nusc, scene):
            if t0 not in pred_by_sample:
                continue
            gt0 = get_gt_by_instance(nusc, t0)
            gt1 = get_gt_by_instance(nusc, t1)
            common = set(gt0) & set(gt1)
            if not common:
                continue

            preds = pred_by_sample[t0]
            gi = list(gt0.keys())
            gp = [gt0[i]['pos'] for i in gi]
            if not preds or not gp:
                continue

            # Hungarian match tracker boxes -> GT at t0
            cost = np.zeros((len(preds), len(gp)))
            for i, p in enumerate(preds):
                pp = np.array(p['translation'][:2])
                for j, g in enumerate(gp):
                    cost[i, j] = np.linalg.norm(pp - g)
            ri, ci = linear_sum_assignment(cost)

            for r, c in zip(ri, ci):
                if cost[r, c] > args.max_dist:
                    continue
                inst = gi[c]
                if inst not in common:
                    continue

                pos = np.array(preds[r]['translation'][:2])
                vel = np.array(preds[r][args.vel_key])
                yaw = gt0[inst]['yaw']  # GT box heading

                # GT speed over the 500ms interval (for grouping)
                gt_speed = np.linalg.norm(gt1[inst]['pos'] - gt0[inst]['pos']) / 0.5
                sg = speed_group_mph(gt_speed)
                if sg not in groups:
                    continue

                for gap in gaps:
                    interp = gap / 500.0
                    gt_target = gt0[inst]['pos'] + (gt1[inst]['pos'] - gt0[inst]['pos']) * interp
                    comp_pos = pos + vel * (gap / 1000.0)
                    delta = comp_pos - gt_target

                    d_lon, d_lat = decompose(delta, yaw)
                    center = np.linalg.norm(delta)

                    acc[gap][sg]['lon'].append(d_lon)
                    acc[gap][sg]['lat'].append(d_lat)
                    acc[gap][sg]['center'].append(center)

    # ---- Aggregate ----
    results = {}
    for gap in gaps:
        results[gap] = {}
        for sg in groups:
            lon = np.abs(np.array(acc[gap][sg]['lon'])) if acc[gap][sg]['lon'] else np.array([0.0])
            lat = np.abs(np.array(acc[gap][sg]['lat'])) if acc[gap][sg]['lat'] else np.array([0.0])
            center = np.array(acc[gap][sg]['center']) if acc[gap][sg]['center'] else np.array([0.0])
            results[gap][sg] = {
                'count': len(acc[gap][sg]['center']),
                'max_center': float(center.max()),
                'mean_center': float(center.mean()),
                'max_lon': float(lon.max()),
                'mean_lon': float(lon.mean()),
                'max_lat': float(lat.max()),
                'mean_lat': float(lat.mean()),
            }

    # ---- Print ----
    print("\n" + "=" * 100)
    print("PHASE 2 LON/LAT DEVIATION - " + args.tracker_name + " (" + args.vel_key + ")")
    print("Deviations are magnitudes (m). Longitudinal = along GT heading, Lateral = perpendicular.")
    print("=" * 100)

    for gap in gaps:
        print("\n--- Gap = " + str(gap) + "ms ---")
        print("{:<10}{:<8}{:<11}{:<12}{:<9}{:<9}{:<9}{:<9}".format(
            'Speed', 'Count', 'MaxCenter', 'MeanCenter', 'MaxLon', 'MeanLon', 'MaxLat', 'MeanLat'))
        print("-" * 77)
        for sg in groups:
            r = results[gap][sg]
            print("{:<10}{:<8}{:<11.4f}{:<12.4f}{:<9.4f}{:<9.4f}{:<9.4f}{:<9.4f}".format(
                sg + ' mph', r['count'], r['max_center'], r['mean_center'],
                r['max_lon'], r['mean_lon'], r['max_lat'], r['mean_lat']))

    # ---- Save ----
    out = {'tracker': args.tracker_name, 'vel_key': args.vel_key,
           'results': {str(g): results[g] for g in gaps}}
    out_path = args.results.rsplit('/', 1)[0] + '/phase2_lonlat_' + args.tracker_name + '.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to " + out_path)


if __name__ == '__main__':
    main()
