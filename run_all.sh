#!/usr/bin/env bash
# Usage: ./run_all.sh <scene> <svo-file>
# Each stage runs in its own conda env. Env names are examples; change them to yours.
set -euo pipefail
SCENE=$1; SVO=$2
ENV_ZED=${ENV_ZED:-zed}          # has pyzed
ENV_DROID=${ENV_DROID:-droidenv}  # DROID-SLAM env
ENV_DOVSG=${ENV_DOVSG:-dovsg}     # ACE + Open3D env
cd "$(dirname "$0")"
conda run --no-capture-output -n "$ENV_ZED"   python 01_export_svo.py --scene "$SCENE" --svo "$SVO"
conda run --no-capture-output -n "$ENV_DROID" python 02_run_droidslam.py --scene "$SCENE"
conda run --no-capture-output -n "$ENV_DOVSG" python 03_align_floor.py --scene "$SCENE"
conda run --no-capture-output -n "$ENV_DOVSG" python 06_build_map.py --scene "$SCENE"
conda run --no-capture-output -n "$ENV_DOVSG" python 08_check_trajectory.py --scene "$SCENE"
conda run --no-capture-output -n "$ENV_DOVSG" python 07_view.py --scene "$SCENE" --html_only
echo "Map + trajectory checked. Open data/$SCENE/viewer.html. Next (needs GPU): 04_train_ace.py then 05_eval_reloc.py"
