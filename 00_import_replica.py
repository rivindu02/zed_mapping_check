"""Import a NICE-SLAM style Replica sequence (rendered RGB-D + exact ground-truth poses) into the project layout.

Replica room/office sequences cover a whole room with revisits, so they test loop closure and drift
against exact ground truth. Rendered data has no sensor noise, so it is an easier case than a real ZED.
Layout: <seq>/results/frame000000.jpg, depth000000.png (uint16, 6553.5 per metre), <seq>/traj.txt (c2w, row-major).
Poses go to poses_zed/ (the reference slot).
"""
import cv2
import numpy as np
from pathlib import Path

import config
from common import scene_args, scene_dir, backproject, valid_mask

p = scene_args(__doc__)
p.add_argument("--seq", required=True, help="e.g. ~/dataset/Replica/nice_slam/Replica/room0")
p.add_argument("--stride", type=int, default=12)
p.add_argument("--max_width", type=int, default=640)
p.add_argument("--max_frames", type=int, default=0)
args = p.parse_args()

seq = Path(args.seq).expanduser()
W, H, FX, FY, CX, CY, SCALE = 1200, 680, 600.0, 600.0, 599.5, 339.5, 6553.5
s = min(1.0, args.max_width / W)
w1, h1 = int(round(W * s)), int(round(H * s))
K = np.array([[FX * s, 0, CX * s], [0, FY * s, CY * s], [0, 0, 1]])

traj = np.loadtxt(seq / "traj.txt").reshape(-1, 4, 4)
root = scene_dir(args.scene)
for sub in ["rgb", "depth", "point", "mask", "calibration", "poses_zed"]:
    (root / sub).mkdir(exist_ok=True)

saved = 0
for i in range(0, len(traj), args.stride):
    bgr = cv2.imread(str(seq / "results" / f"frame{i:06d}.jpg"))
    d = cv2.imread(str(seq / "results" / f"depth{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / SCALE
    if bgr is None or d is None:
        continue
    bgr = cv2.resize(bgr, (w1, h1), interpolation=cv2.INTER_AREA)
    d = cv2.resize(d, (w1, h1), interpolation=cv2.INTER_NEAREST)
    name = f"{saved:06d}"
    cv2.imwrite(str(root / "rgb" / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.save(root / "depth" / f"{name}.npy", d * 1000.0)
    np.save(root / "point" / f"{name}.npy", backproject(d, K))
    np.save(root / "mask" / f"{name}.npy", valid_mask(d))
    np.savetxt(root / "calibration" / f"{name}.txt", K)
    # traj.txt of this Replica copy is already camera-to-world with OpenCV axes (checked against DROID-SLAM:
    # relative-rotation error 0.07 deg; flipping y and z, as some NICE-SLAM loaders do, gave 31 deg).
    np.savetxt(root / "poses_zed" / f"{name}.txt", traj[i])
    saved += 1
    if args.max_frames and saved >= args.max_frames:
        break
np.savetxt(root / "calib.txt", [[K[0, 0], K[1, 1], K[0, 2], K[1, 2]]])
print(f"Saved {saved} frames ({w1}x{h1}) to {root}")
