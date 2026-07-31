"""
Phase 2 IoU: BEV and 3D IoU between compensated boxes and GT target boxes
=========================================================================

For each matched (tracker, GT) pair after time compensation:
  - Build the COMPENSATED box: tracker box center shifted by velocity*gap,
    keeping tracker's size (wlh) and heading (yaw).
  - Build the GT TARGET box: GT size + GT heading, centered at the
    interpolated GT position (between t and t+500ms).
  - Compute BEV IoU (top-down rotated rectangle overlap) and
    3D IoU (BEV overlap x height overlap / union volume).

Reports mean BEV IoU and mean 3D IoU per (temporal gap x speed group).

Speed groups (mph): 0-20, 20-40, 40-60
Temporal gaps (ms): 200, 300, 400, 500

Requires: shapely (for rotated-rectangle intersection)

Usage:
  python phase2_iou.py \
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
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate


MPH = 2.23694


def quaternion_to_yaw(q):
    return Quaternion(q).yaw_pitch_roll[0]


def speed_group_mph(speed_mps):
    mph = speed_mps * MPH
    if mph < 20:
        return '0-20'
    elif mph < 40:
        return '20-40'
    elif mph < 60:
        return '40-60'
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
    """GT indexed by instance_token: position(xy), z, size(wlh), yaw."""
    sample = nusc.get('sample', sample_token)
    out = {}
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        out[ann['instance_token']] = {
            'pos': np.array(ann['translation'][:2]),
            'z': ann['translation'][2],
            'wlh': np.array(ann['size']),  # nuScenes size = [w, l, h]
            'yaw': quaternion_to_yaw(ann['rotation']),
        }
    return out


def make_bev_polygon(cx, cy, w, l, yaw):
    """Axis-aligned rectangle (length l along x, width w along y), then rotate by yaw."""
    poly = Polygon([(-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)])
    poly = rotate(poly, np.degrees(yaw), origin=(0, 0))
    poly = translate(poly, cx, cy)
    return poly


def bev_iou(boxA, boxB):
    """boxes = (cx, cy, w, l, yaw). Returns (iou_bev, inter_area, areaA, areaB)."""
    pa = make_bev_polygon(*boxA)
    pb = make_bev_polygon(*boxB)
    if not pa.is_valid or not pb.is_valid:
        return 0.0, 0.0, pa.area, pb.area
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    iou = inter / union if union > 0 else 0.0
    return iou, inter, pa.area, pb.area


def iou_3d(boxA_bev, hA, zA, boxB_bev, hB, zB):
    """3D IoU using BEV intersection x height overlap / union volume."""
    _, inter_area, areaA, areaB = bev_iou(boxA_bev, boxB_bev)
    # height overlap along z
    top = min(zA + hA / 2, zB + hB / 2)
    bot = max(zA - hA / 2, zB - hB / 2)
    h_overlap = max(0.0, top - bot)
    inter_vol = inter_area * h_overlap
    volA = areaA * hA
    volB = areaB * hB
    union_vol = volA + volB - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0


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

    acc = {g: {sg: {'bev': [], '3d': []} for sg in groups} for g in gaps}

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

                pred = preds[r]
                pos = np.array(pred['translation'][:2])
                z_pred = pred['translation'][2]
                vel = np.array(pred[args.vel_key])
                # tracker box size: nuScenes result format size = [w, l, h]
                w_pred, l_pred, h_pred = pred['size'][0], pred['size'][1], pred['size'][2]
                yaw_pred = quaternion_to_yaw(pred['rotation']) if 'rotation' in pred \
                    else quaternion_to_yaw(pred.get('rotation_kalman_cv', [1, 0, 0, 0]))

                # GT speed for grouping
                gt_speed = np.linalg.norm(gt1[inst]['pos'] - gt0[inst]['pos']) / 0.5
                sg = speed_group_mph(gt_speed)
                if sg not in groups:
                    continue

                gt_wlh = gt0[inst]['wlh']  # [w, l, h]
                gt_yaw = gt0[inst]['yaw']
                gt_z = gt0[inst]['z']

                for gap in gaps:
                    interp = gap / 500.0
                    gt_target_xy = gt0[inst]['pos'] + (gt1[inst]['pos'] - gt0[inst]['pos']) * interp
                    comp_xy = pos + vel * (gap / 1000.0)

                    comp_box = (comp_xy[0], comp_xy[1], w_pred, l_pred, yaw_pred)
                    gt_box = (gt_target_xy[0], gt_target_xy[1], gt_wlh[0], gt_wlh[1], gt_yaw)

                    iou_bev, _, _, _ = bev_iou(comp_box, gt_box)
                    iou3 = iou_3d(comp_box, h_pred, z_pred, gt_box, gt_wlh[2], gt_z)

                    acc[gap][sg]['bev'].append(iou_bev)
                    acc[gap][sg]['3d'].append(iou3)

    # Aggregate
    results = {}
    for gap in gaps:
        results[gap] = {}
        for sg in groups:
            bev = np.array(acc[gap][sg]['bev']) if acc[gap][sg]['bev'] else np.array([0.0])
            d3 = np.array(acc[gap][sg]['3d']) if acc[gap][sg]['3d'] else np.array([0.0])
            results[gap][sg] = {
                'count': len(acc[gap][sg]['bev']),
                'mean_bev_iou': float(bev.mean()),
                'median_bev_iou': float(np.median(bev)),
                'mean_3d_iou': float(d3.mean()),
                'median_3d_iou': float(np.median(d3)),
            }

    print("\n" + "=" * 90)
    print("PHASE 2 IoU - " + args.tracker_name + " (" + args.vel_key + ")")
    print("IoU between COMPENSATED box and GT target box. Higher is better.")
    print("=" * 90)
    for gap in gaps:
        print("\n--- Gap = " + str(gap) + "ms ---")
        print("{:<10}{:<8}{:<14}{:<14}{:<14}{:<14}".format(
            'Speed', 'Count', 'MeanBEV_IoU', 'MedBEV_IoU', 'Mean3D_IoU', 'Med3D_IoU'))
        print("-" * 74)
        for sg in groups:
            r = results[gap][sg]
            print("{:<10}{:<8}{:<14.4f}{:<14.4f}{:<14.4f}{:<14.4f}".format(
                sg + ' mph', r['count'], r['mean_bev_iou'], r['median_bev_iou'],
                r['mean_3d_iou'], r['median_3d_iou']))

    out = {'tracker': args.tracker_name, 'vel_key': args.vel_key,
           'results': {str(g): results[g] for g in gaps}}
    out_path = args.results.rsplit('/', 1)[0] + '/phase2_iou_' + args.tracker_name + '.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to " + out_path)


if __name__ == '__main__':
    main()
