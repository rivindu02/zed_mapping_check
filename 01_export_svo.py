"""ZED SVO -> DovSG-style folder (rgb jpg, depth npy in mm, point npy, mask npy, calibration).

Needs the ZED SDK and pyzed (use a separate env from the DROID-SLAM env).
Also saves the ZED positional-tracking poses to poses_zed/ as an independent reference trajectory.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from common import scene_args, scene_dir, backproject, valid_mask

p = scene_args(__doc__)
p.add_argument("--svo", required=True)
p.add_argument("--depth_mode", default="NEURAL", choices=["NEURAL", "ULTRA", "QUALITY", "PERFORMANCE"])
p.add_argument("--stride", type=int, default=config.STRIDE)
p.add_argument("--max_frames", type=int, default=0, help="0 = all")
args = p.parse_args()

try:
    import pyzed.sl as sl
except ImportError:
    sys.exit("pyzed not found. Install the ZED SDK and run: python get_python_api.py")

root = scene_dir(args.scene)
for sub in ["rgb", "depth", "point", "mask", "calibration", "poses_zed"]:
    (root / sub).mkdir(exist_ok=True)

init = sl.InitParameters()
init.set_from_svo_file(args.svo)
init.svo_real_time_mode = False
init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
init.coordinate_units = sl.UNIT.MILLIMETER
init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE   # x right, y down, z forward (OpenCV)

zed = sl.Camera()
status = zed.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    sys.exit(f"Cannot open SVO: {status}")
zed.enable_positional_tracking(sl.PositionalTrackingParameters())

cal = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
fx, fy, cx, cy = cal.fx, cal.fy, cal.cx, cal.cy
total = zed.get_svo_number_of_frames()
print(f"SVO frames: {total}, left cam fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

img, dep, pose = sl.Mat(), sl.Mat(), sl.Pose()
rt = sl.RuntimeParameters()
saved, i = 0, 0
K0 = None
while True:
    err = zed.grab(rt)
    if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        break
    if err != sl.ERROR_CODE.SUCCESS:
        continue
    # Tracking runs on every grabbed frame; only every stride-th frame is saved.
    if i % args.stride == 0:
        zed.retrieve_image(img, sl.VIEW.LEFT)
        zed.retrieve_measure(dep, sl.MEASURE.DEPTH)
        bgr = np.ascontiguousarray(img.get_data()[:, :, :3])
        depth_mm = np.nan_to_num(dep.get_data().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        h0, w0 = depth_mm.shape
        s = min(1.0, config.MAX_WIDTH / w0)
        if s < 1.0:
            w1, h1 = int(round(w0 * s)), int(round(h0 * s))
            bgr = cv2.resize(bgr, (w1, h1), interpolation=cv2.INTER_AREA)
            depth_mm = cv2.resize(depth_mm, (w1, h1), interpolation=cv2.INTER_NEAREST)
        K = np.array([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]])
        K0 = K
        name = f"{saved:06d}"
        depth_m = depth_mm / 1000.0
        mask = valid_mask(depth_m)
        pts = backproject(depth_m, K)
        cv2.imwrite(str(root / "rgb" / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(root / "depth" / f"{name}.npy", depth_mm)
        np.save(root / "point" / f"{name}.npy", pts)
        np.save(root / "mask" / f"{name}.npy", mask)
        np.savetxt(root / "calibration" / f"{name}.txt", K)
        state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        if state == sl.POSITIONAL_TRACKING_STATE.OK:
            np.savetxt(root / "poses_zed" / f"{name}.txt", np.array(pose.pose_data().m))
        saved += 1
        if args.max_frames and saved >= args.max_frames:
            break
    i += 1
zed.close()

np.savetxt(root / "calib.txt", [[K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]]])
print(f"Saved {saved} frames to {root}")
