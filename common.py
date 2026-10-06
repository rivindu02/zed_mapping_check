"""Small helpers shared by the numbered scripts. Pure numpy, no DovSG imports."""
import argparse
import json
from pathlib import Path

import numpy as np

import config


def scene_args(description=""):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--scene", required=True, help="scene name, data goes to data/<scene>/")
    return p


def scene_dir(name):
    d = config.DATA_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def frame_names(root):
    return sorted(p.stem for p in (Path(root) / "rgb").glob("*.jpg"))


def load_K(root, name):
    return np.loadtxt(Path(root) / "calibration" / f"{name}.txt")


def load_depth_m(root, name):
    return np.load(Path(root) / "depth" / f"{name}.npy").astype(np.float32) / 1000.0


def load_pose(root, name, folder="poses"):
    return np.loadtxt(Path(root) / folder / f"{name}.txt")


def load_poses(root, folder="poses"):
    return {n: load_pose(root, n, folder) for n in frame_names(root)
            if (Path(root) / folder / f"{n}.txt").exists()}


def backproject(depth_m, K, mask=None):
    """Camera-frame points (H,W,3) from a metric depth map."""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def valid_mask(depth_m):
    return np.isfinite(depth_m) & (depth_m > config.DEPTH_MIN) & (depth_m < config.DEPTH_MAX)


def to_world(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
