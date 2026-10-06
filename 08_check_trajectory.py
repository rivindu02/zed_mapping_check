"""ATE of the DROID-SLAM trajectory against the reference (ZED tracking or TUM ground truth), plus loop-closure gap.

Rigid (no scale) alignment, because both trajectories are metric. Also reports end-to-start distance, which is a
useful loop-closure check when the scan returns to its starting point.
"""
import numpy as np

import config
from common import scene_args, scene_dir, frame_names, write_json

args = scene_args(__doc__).parse_args()
root = scene_dir(args.scene)
names = [n for n in frame_names(root)
         if (root / "poses_droidslam" / f"{n}.txt").exists() and (root / "poses_zed" / f"{n}.txt").exists()]
assert len(names) > 10, "Need poses_droidslam/ and poses_zed/ for the same frames"
B = np.stack([np.loadtxt(root / "poses_droidslam" / f"{n}.txt")[:3, 3] for n in names])
A = np.stack([np.loadtxt(root / "poses_zed" / f"{n}.txt")[:3, 3] for n in names])
ma, mb = A.mean(0), B.mean(0)
U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
R = Vt.T @ D @ U.T
err = np.linalg.norm((R @ (A - ma).T).T + mb - B, axis=1)
path_len = float(np.sum(np.linalg.norm(np.diff(B, axis=0), axis=1)))
res = {
    "frames": len(names), "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))), "ate_max_m": float(err.max()),
    "path_length_m": path_len, "end_to_start_gap_m": float(np.linalg.norm(B[-1] - B[0])),
    "pass_ate": bool(np.sqrt(np.mean(err ** 2)) < config.PASS_ATE_M),
}
write_json(config.REPORT_DIR / f"{args.scene}_trajectory.json", res)
print(res)
