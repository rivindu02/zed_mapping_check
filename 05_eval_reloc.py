"""Relocalise held-out frames with ACE (+PnP RANSAC), then refine with multi-scale colored ICP.

Ground truth is the floor-aligned DROID-SLAM pose. The ICP target is a map fused from the TRAIN frames only,
so a held-out frame never sees itself. ICP settings follow DovSG's calculate_alignment_colored_icp.
Run in the DovSG env.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

import config
from common import (scene_args, scene_dir, frame_names, load_K, load_depth_m, load_pose,
                    backproject, valid_mask, to_world, rot_angle_deg, write_json)

args = scene_args(__doc__).parse_args()
root = scene_dir(args.scene)
held = (root / "heldout.txt").read_text().split()
names = [n for n in frame_names(root) if (root / "poses" / f"{n}.txt").exists()]
train = [n for n in names if n not in set(held)]


def frame_cloud(n, pose):
    d = load_depth_m(root, n)
    m = valid_mask(d)
    bgr = cv2.imread(str(root / "rgb" / f"{n}.jpg"))
    pts = to_world(backproject(d, load_K(root, n))[m], pose)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(bgr[m][:, ::-1] / 255.0)
    return pcd


# target map from train frames only
target = o3d.geometry.PointCloud()
for n in train[:: config.MAP_FRAME_INTERVAL]:
    target += frame_cloud(n, load_pose(root, n)).voxel_down_sample(0.01)
target = target.voxel_down_sample(0.01)

# ACE rough poses
obs = {}
for n in held:
    bgr = cv2.imread(str(root / "rgb" / f"{n}.jpg"))
    obs[n] = {"rgb": (bgr[:, :, ::-1] / 255.0).astype(np.float32), "intrinsic": load_K(root, n)}
os.chdir(config.DOVSG_ROOT)
sys.path.insert(0, str(config.DOVSG_ROOT))
from ace.test_ace import test_ace
est = test_ace(str((root / "ace" / "ace.pt").resolve()), obs)
os.chdir(config.HERE)


def colored_icp(source, target):
    T = np.eye(4)
    ok = False
    for i, (r, it) in enumerate(zip([0.04, 0.02, 0.01], [50, 30, 14])):
        s, t = source.voxel_down_sample(r), target.voxel_down_sample(r)
        for c in (s, t):
            c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=r * 3, max_nn=30))
        try:
            res = o3d.pipelines.registration.registration_colored_icp(
                s, t, r, T,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.9999),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=it))
        except Exception:
            continue
        if len(res.correspondence_set) >= 0.2 * min(len(s.points), len(t.points)):
            T = res.transformation
            ok = i == 2
    return ok, T


rows = []
for n, (pose_ace, inl) in zip(held, est):
    gt = load_pose(root, n)
    pose_ace = np.asarray(pose_ace, dtype=np.float64)
    src = frame_cloud(n, pose_ace)
    # crop the target around the ACE estimate to keep ICP fast
    c = pose_ace[:3, 3]
    tgt = target.crop(o3d.geometry.AxisAlignedBoundingBox(c - 3.0, c + 3.0))
    ok, Ticp = colored_icp(src, tgt) if len(tgt.points) > 1000 else (False, np.eye(4))
    pose_icp = Ticp @ pose_ace if ok else pose_ace
    rows.append({
        "frame": n, "ace_inliers": float(inl),
        "ace_trans_m": float(np.linalg.norm(pose_ace[:3, 3] - gt[:3, 3])),
        "ace_rot_deg": rot_angle_deg(pose_ace[:3, :3], gt[:3, :3]),
        "icp_ok": bool(ok),
        "icp_trans_m": float(np.linalg.norm(pose_icp[:3, 3] - gt[:3, 3])),
        "icp_rot_deg": rot_angle_deg(pose_icp[:3, :3], gt[:3, :3]),
    })


def frac(k_t, k_r):
    return float(np.mean([r[k_t] < config.PASS_TRANS_M and r[k_r] < config.PASS_ROT_DEG for r in rows]))


summary = {
    "n_heldout": len(rows),
    "ace_within_5cm_5deg": frac("ace_trans_m", "ace_rot_deg"),
    "ace_icp_within_5cm_5deg": frac("icp_trans_m", "icp_rot_deg"),
    "ace_median_trans_m": float(np.median([r["ace_trans_m"] for r in rows])),
    "ace_icp_median_trans_m": float(np.median([r["icp_trans_m"] for r in rows])),
    "icp_success_rate": float(np.mean([r["icp_ok"] for r in rows])),
}
write_json(config.REPORT_DIR / f"{args.scene}_reloc.json", {"summary": summary, "frames": rows})
print(summary)
