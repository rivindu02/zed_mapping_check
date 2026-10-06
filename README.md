# zed_mapping_check

A standalone check of the DovSG mapping idea on a ZED 2i recording: DROID-SLAM poses, floor alignment, an ACE scene-coordinate model, colored ICP refinement, and a fused 1 cm voxel map you can look at together with the camera trajectory.

This project is deliberately separate from DovSG. It does not import the controller, the perception models, the scene graph, or the planner. It only reads three things from `/home/rivindu02/Documents/FYP/DovSG` (set in `config.py`): the DROID-SLAM submodule and weights, and the vendored `ace/` package. Nothing there is modified.

Colored ICP is not part of DovSG's mapping stage. DovSG uses it at relocalization time, to refine the ACE pose against the stored map. Here it is checked in `05_eval_reloc.py`.

## Status

All scripts pass a syntax check. None has been run end to end yet: this machine has no GPU, no conda, no Open3D, no ZED SDK, and the DovSG checkpoints and DROID-SLAM submodule are not present. Expect small fixes on the first real run, mostly in the SVO export (pyzed version differences) and in DROID-SLAM's import path.

## Environments

| Stage | Env | Needs |
|---|---|---|
| 01 export | `zed` | ZED SDK, `pyzed`, opencv, numpy |
| 02 DROID-SLAM | DovSG's `droidenv` (DovSG's code calls it `droidslam`) | CUDA GPU, DROID-SLAM built, `droid.pth`, scipy |
| 03, 06, 07, 08 | `dovsg` or any env with open3d, numpy, scipy, opencv | CPU is enough |
| 04, 05 ACE | `dovsg` | CUDA GPU, `dsacstar` built, `ace_encoder_pretrained.pt` |

## Run order

```
python 01_export_svo.py --scene room1 --svo path/to/file.svo2     # or: python 00_import_rgbd.py --scene tum1 --tum <TUM folder>
python 02_run_droidslam.py --scene room1
python 03_align_floor.py   --scene room1
python 06_build_map.py     --scene room1
python 08_check_trajectory.py --scene room1
python 07_view.py          --scene room1          # Open3D window + data/room1/viewer.html
python 04_train_ace.py     --scene room1          # GPU
python 05_eval_reloc.py    --scene room1          # GPU
```

`./run_all.sh room1 file.svo2` runs the first six in order.

Outputs are in `data/<scene>/` (frames, poses, `map_1cm.ply`, `viewer.html`, `ace/ace.pt`) and `reports/` (JSON metrics).

## Choosing a recording

No public full-room ZED 2i SVO was found. Stereolabs offers short demo SVOs (indoor sofa and wall, `download.stereolabs.com/assets/svo_samples/sofa` and `/wall`), which are only good for testing the exporter. Two practical routes:

1. Record your own with the ZED 2i: one slow loop around a furnished room, 60 to 90 seconds, ending where you started, floor in view, no fast turns, good light, some texture on the walls. This also gives the loop-closure check.
2. To test the method without a ZED, convert a public RGB-D sequence with `00_import_rgbd.py` (for example TUM `fr3/long_office_household`).

## When to stop (stopping criteria)

The stage is done when all of these pass on one scene. Numbers are written to `reports/`.

| Check | Pass | If it fails |
|---|---|---|
| DROID-SLAM completes | pose count equals frame count, no NaN | fix depth units, env, calibration, rerun |
| Trajectory vs reference (`08`) | ATE RMSE under 5 cm | rescan slower, more texture and overlap |
| Floor alignment (`03`) | floor z std under 2 cm, tilt under 25 degrees | redo the fit, make sure the floor is visible in the scan |
| Fused map (`07`) | walls are single-layer, loop closes near the start | treat as drift and rescan |
| ACE training (`04`) | finishes, `ace.pt` written | check frame, pose and calibration counts |
| ACE held-out (`05`) | at least 80 to 90 percent of held-out frames within 5 cm and 5 degrees | more views, smaller stride, rescan |
| Colored ICP (`05`) | median error not worse than ACE alone, success rate high | tune voxel radii or the crop size |

Hard stop: if two full rescans fail the trajectory or loop-closure check, the cause is the sensor or scene (low texture, glare, fast motion, noisy ZED depth), not the method. Change the capture or the depth mode instead of tuning parameters. Soft stop: once everything passes once, record the numbers and move on. Do not chase sub-centimetre accuracy, since the map is 1 cm and ACE is coarse by design.

## Known limits

- ZED depth is stereo or neural, noisier than RealSense at edges and range. `DEPTH_MIN` and `DEPTH_MAX` in `config.py` may need tuning.
- DROID-SLAM here runs in RGB-D mode on the left image only. The right image and the IMU are unused.
- The ZED tracking poses are a reference, not ground truth. Differences can come from either side.
- ACE held-out frames come from the same recording, so the score measures interpolation inside the scanned area, not relocalization after the scene changes.
