"""Shared settings for the standalone mapping check.

Nothing in DOVSG_ROOT is modified. Only DROID-SLAM, its weights and the vendored ACE package are read from there.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOVSG_ROOT = Path(os.environ.get("DOVSG_ROOT", "/home/rivindu02/Documents/FYP/DovSG"))

DROID_PKG = Path(os.environ.get("DROID_PKG", DOVSG_ROOT / "third_party" / "DROID-SLAM" / "droid_slam"))
DROID_WEIGHTS = Path(os.environ.get("DROID_WEIGHTS", DOVSG_ROOT / "checkpoints" / "droid-slam" / "droid.pth"))

DATA_DIR = HERE / "data"
REPORT_DIR = HERE / "reports"

# Depth validity window in metres. ZED stereo depth is noisy far away; tighten if the map looks fuzzy.
DEPTH_MIN = 0.3
DEPTH_MAX = 5.0

# Export: keep every STRIDE-th frame of the SVO. Pick it so neighbouring frames overlap about 80-90 percent.
STRIDE = 5
MAX_WIDTH = 1280          # downscale saved frames to at most this width

# Mapping
MAP_VOXEL = 0.01          # metres, same as DovSG's resolution
MAP_FRAME_INTERVAL = 3    # fuse every 3rd frame, same as DovSG demo.py
VIEW_VOXEL = 0.05

# Held-out split for the ACE evaluation: every HOLDOUT_EVERY-th frame is excluded from training.
HOLDOUT_EVERY = 10

# Pass thresholds used by the stopping criteria
PASS_TRANS_M = 0.05
PASS_ROT_DEG = 5.0
PASS_ATE_M = 0.05
