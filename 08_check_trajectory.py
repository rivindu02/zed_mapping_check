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

# Loop-closure check that does not need the scan to end where it started: find revisits (close in space, similar
# viewing direction, far apart in time) using the reference poses, then compare the relative pose between the two
# visits in the DROID trajectory with the same relative pose in the reference. Drift shows up as a large difference.
P = [np.loadtxt(root / "poses_zed" / f"{n}.txt") for n in names]
Q = [np.loadtxt(root / "poses_droidslam" / f"{n}.txt") for n in names]
min_gap = max(10, len(names) // 5)
pairs = []
for i in range(len(names)):
    for j in range(i + min_gap, len(names)):
        d = np.linalg.norm(P[i][:3, 3] - P[j][:3, 3])
        ang = np.degrees(np.arccos(np.clip(P[i][:3, 2] @ P[j][:3, 2], -1, 1)))
        if d < 0.5 and ang < 40:
            pairs.append((i, j))
if pairs:
    te, re_ = [], []
    for i, j in pairs:
        Rel_ref = np.linalg.inv(P[i]) @ P[j]
        Rel_est = np.linalg.inv(Q[i]) @ Q[j]
        dT = np.linalg.inv(Rel_ref) @ Rel_est
        te.append(float(np.linalg.norm(dT[:3, 3])))
        re_.append(float(np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))))
    res.update({
        "revisit_pairs": len(pairs), "revisit_trans_err_median_m": float(np.median(te)),
        "revisit_trans_err_max_m": float(np.max(te)), "revisit_rot_err_median_deg": float(np.median(re_)),
        "pass_revisit": bool(np.median(te) < config.PASS_TRANS_M),
    })
else:
    res["revisit_pairs"] = 0
write_json(config.REPORT_DIR / f"{args.scene}_trajectory.json", res)
print(res)
