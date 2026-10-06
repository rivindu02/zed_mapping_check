"""Fuse masked depth points with the floor-aligned poses into a 1 cm voxel map and save PLY files + trajectory."""
import cv2
import numpy as np
import open3d as o3d

import config
from common import (scene_args, scene_dir, frame_names, load_K, load_depth_m, load_pose,
                    backproject, valid_mask, to_world)

args = scene_args(__doc__).parse_args()
root = scene_dir(args.scene)
names = [n for n in frame_names(root) if (root / "poses" / f"{n}.txt").exists()]

acc = o3d.geometry.PointCloud()
for n in names[:: config.MAP_FRAME_INTERVAL]:
    d = load_depth_m(root, n)
    m = valid_mask(d)
    bgr = cv2.imread(str(root / "rgb" / f"{n}.jpg"))
    pts = to_world(backproject(d, load_K(root, n))[m], load_pose(root, n))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.colors = o3d.utility.Vector3dVector(bgr[m][:, ::-1] / 255.0)
    acc += pc.voxel_down_sample(config.MAP_VOXEL)
acc = acc.voxel_down_sample(config.MAP_VOXEL)
acc, _ = acc.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.5)   # same values as DovSG's defaults

o3d.io.write_point_cloud(str(root / "map_1cm.ply"), acc)
o3d.io.write_point_cloud(str(root / "map_view.ply"), acc.voxel_down_sample(config.VIEW_VOXEL))
np.save(root / "trajectory.npy", np.stack([load_pose(root, n) for n in names]))
print(f"{len(acc.points)} voxels at {config.MAP_VOXEL} m -> {root / 'map_1cm.ply'}")
