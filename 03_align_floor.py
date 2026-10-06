"""Floor alignment without any detector.

The camera y-axis points down, so the mean camera y-axis in the DROID world frame is an estimate of 'down'.
Take the lowest 20 percent of points along 'up', fit a plane with RANSAC, rotate its normal to +z and
shift the floor to z = 0. Writes poses/ and reports/<scene>_floor.json.
"""
import numpy as np
import open3d as o3d

import config
from common import (scene_args, scene_dir, frame_names, load_K, load_depth_m, load_pose,
                    backproject, valid_mask, to_world, write_json)

args = scene_args(__doc__).parse_args()
root = scene_dir(args.scene)
names = [n for n in frame_names(root) if (root / "poses_droidslam" / f"{n}.txt").exists()]
poses = {n: load_pose(root, n, "poses_droidslam") for n in names}

down = np.mean([poses[n][:3, 1] for n in names], axis=0)
down /= np.linalg.norm(down)
up = -down

pts = []
for n in names[:: max(1, len(names) // 60)]:
    d = load_depth_m(root, n)
    m = valid_mask(d)
    p = backproject(d, load_K(root, n))[m][::17]
    pts.append(to_world(p, poses[n]))
pts = np.vstack(pts)
h = pts @ up
low = pts[h <= np.percentile(h, 20)]

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low))
(a, b, c, dd), inl = pcd.segment_plane(distance_threshold=0.03, ransac_n=3, num_iterations=2000)
n_vec = np.array([a, b, c])
n_vec /= np.linalg.norm(n_vec)
if n_vec @ up < 0:
    n_vec, dd = -n_vec, -dd
tilt = float(np.degrees(np.arccos(np.clip(n_vec @ up, -1, 1))))

axis = np.cross(n_vec, [0, 0, 1])
s = np.linalg.norm(axis)
R = np.eye(3) if s < 1e-8 else o3d.geometry.get_rotation_matrix_from_axis_angle(
    axis / s * np.arctan2(s, n_vec[2]))
floor_pts = np.asarray(pcd.points)[inl] @ R.T
T = np.eye(4)
T[:3, :3] = R
T[2, 3] = -floor_pts[:, 2].mean()

out = root / "poses"
out.mkdir(exist_ok=True)
for n in names:
    np.savetxt(out / f"{n}.txt", T @ poses[n])

report = {
    "floor_inliers": int(len(inl)), "candidate_points": int(len(low)),
    "floor_z_std_m": float(floor_pts[:, 2].std()),
    "tilt_vs_camera_up_deg": tilt,
    "pass": bool(floor_pts[:, 2].std() < 0.02 and tilt < 25),
}
write_json(config.REPORT_DIR / f"{args.scene}_floor.json", report)
print(report)
