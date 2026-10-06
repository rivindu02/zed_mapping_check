"""Method sanity check without a ZED: convert a TUM RGB-D sequence to the same folder layout.

Download e.g. rgbd_dataset_freiburg3_long_office_household from
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download and pass its folder.
Uses the freiburg3 intrinsics. Ground-truth poses are written to poses_zed/ (the reference slot).
"""
import cv2
import numpy as np
from pathlib import Path

import config
from common import scene_args, scene_dir, backproject, valid_mask

p = scene_args(__doc__)
p.add_argument("--tum", required=True, help="path to the extracted TUM sequence folder")
p.add_argument("--stride", type=int, default=3)
p.add_argument("--max_frames", type=int, default=0)
args = p.parse_args()

FX, FY, CX, CY, DEPTH_FACTOR = 535.4, 539.2, 320.1, 247.6, 5000.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
tum = Path(args.tum)


def read_list(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        t, *rest = line.split()
        rows.append((float(t), rest))
    return rows


rgb_l, dep_l, gt_l = read_list(tum / "rgb.txt"), read_list(tum / "depth.txt"), read_list(tum / "groundtruth.txt")
dep_t = np.array([t for t, _ in dep_l])
gt_t = np.array([t for t, _ in gt_l])

root = scene_dir(args.scene)
for sub in ["rgb", "depth", "point", "mask", "calibration", "poses_zed"]:
    (root / sub).mkdir(exist_ok=True)


def quat_pose(v):
    from scipy.spatial.transform import Rotation
    x, y, z, qx, qy, qz, qw = map(float, v)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


saved = 0
for k, (t, (rgb_rel,)) in enumerate(rgb_l):
    if k % args.stride:
        continue
    j = int(np.argmin(np.abs(dep_t - t)))
    g = int(np.argmin(np.abs(gt_t - t)))
    if abs(dep_t[j] - t) > 0.02 or abs(gt_t[g] - t) > 0.02:
        continue
    bgr = cv2.imread(str(tum / rgb_rel))
    d_raw = cv2.imread(str(tum / dep_l[j][1][0]), cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_m = d_raw / DEPTH_FACTOR
    name = f"{saved:06d}"
    cv2.imwrite(str(root / "rgb" / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(root / "depth" / f"{name}.npy", depth_m * 1000.0)
    np.save(root / "point" / f"{name}.npy", backproject(depth_m, K))
    np.save(root / "mask" / f"{name}.npy", valid_mask(depth_m))
    np.savetxt(root / "calibration" / f"{name}.txt", K)
    np.savetxt(root / "poses_zed" / f"{name}.txt", quat_pose(gt_l[g][1]))
    saved += 1
    if args.max_frames and saved >= args.max_frames:
        break
np.savetxt(root / "calib.txt", [[FX, FY, CX, CY]])
print(f"Saved {saved} frames to {root}")
