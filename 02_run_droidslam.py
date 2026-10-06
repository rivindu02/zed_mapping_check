"""Run DROID-SLAM (RGB-D mode) on the exported frames and write poses_droidslam/ (4x4 camera-to-world).

Run inside the DROID-SLAM conda env. The logic follows DovSG's pose_estimation.py (about 320x240 input,
sides cropped to multiples of 8) but prints all output so failures are visible.
"""
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import config
from common import frame_names, load_K

sys.path.insert(0, str(config.DROID_PKG))


def image_stream(root, names):
    for t, n in enumerate(names):
        K = load_K(root, n)
        color = cv2.imread(str(root / "rgb" / f"{n}.jpg"))
        depth = np.load(root / "depth" / f"{n}.npy").astype(np.float32) / 1000.0
        h0, w0, _ = color.shape
        h1 = int(h0 * np.sqrt((240 * 320) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((240 * 320) / (h0 * w0)))
        color = cv2.resize(color, (w1, h1))[: h1 - h1 % 8, : w1 - w1 % 8]
        color = torch.as_tensor(color).permute(2, 0, 1)
        d = torch.from_numpy(depth).float()[None, None]
        d = F.interpolate(d, (h1, w1), mode="nearest").squeeze()[: h1 - h1 % 8, : w1 - w1 % 8]
        intr = torch.as_tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
        intr[0::2] *= w1 / w0
        intr[1::2] *= h1 / h0
        yield t, color[None], d, intr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--weights", default=str(config.DROID_WEIGHTS))
    ap.add_argument("--buffer", type=int, default=2048)
    a = ap.parse_args()

    from droid import Droid  # imported late so the path insert above is active

    root = config.DATA_DIR / a.scene
    names = frame_names(root)
    assert names, f"No frames in {root}/rgb"
    med = np.median(np.load(root / "depth" / f"{names[0]}.npy")[np.load(root / "mask" / f"{names[0]}.npy")])
    assert 300 < med < 15000, f"Median depth {med} is not in millimetres. Check the exporter."

    args = argparse.Namespace(
        weights=a.weights, buffer=a.buffer, disable_vis=True, stereo=False, upsample=True,
        beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0,
        frontend_window=25, frontend_radius=2, frontend_nms=1,
        backend_thresh=22.0, backend_radius=2, backend_nms=3, t0=0, image_size=None,
    )
    droid = None
    for t, image, depth, intr in tqdm(image_stream(root, names), total=len(names), desc="DROID-SLAM"):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        droid.track(t, image, depth=depth, intrinsics=intr)
    # (N, 7) x y z qx qy qz qw, camera-to-world. Upstream DROID-SLAM's trajectory filler reads (t, image, intrinsics);
    # DovSG's variant also reads depth. terminate() consumes state, so pick the form before calling it.
    import inspect
    wants_depth = "depth" in inspect.getsource(type(droid.traj_filler).__call__)
    if wants_depth:
        traj = droid.terminate(image_stream(root, names))
    else:
        traj = droid.terminate((t, im, k) for t, im, _, k in image_stream(root, names))

    out = root / "poses_droidslam"
    out.mkdir(exist_ok=True)
    for i, q in enumerate(traj):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(q[3:7]).as_matrix()
        T[:3, 3] = q[:3]
        np.savetxt(out / f"{names[i]}.txt", T)
    print(f"Wrote {len(traj)} poses for {len(names)} frames to {out}")
    assert len(traj) == len(names), "Pose count differs from frame count"


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
