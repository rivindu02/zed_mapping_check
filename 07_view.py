"""View the voxel map and the camera trajectory.

Open3D window: map, camera frustums, trajectory line (green start, red end), world axes.
Keys: T toggle trajectory, F toggle frustums, Z toggle reference (ZED/GT) trajectory.
Also writes data/<scene>/viewer.html, a self-contained three.js page that needs no GPU or Open3D.
"""
import base64
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from common import scene_args, scene_dir, frame_names

p = scene_args(__doc__)
p.add_argument("--html_only", action="store_true")
p.add_argument("--frustum_every", type=int, default=10)
args = p.parse_args()
root = scene_dir(args.scene)

pcd = o3d.io.read_point_cloud(str(root / "map_view.ply"))
names = [n for n in frame_names(root) if (root / "poses" / f"{n}.txt").exists()]
traj = np.stack([np.loadtxt(root / "poses" / f"{n}.txt") for n in names])
ref_names = [n for n in names if (root / "poses_zed" / f"{n}.txt").exists()]
ref = np.stack([np.loadtxt(root / "poses_zed" / f"{n}.txt") for n in ref_names]) if ref_names else None

# ZED/GT reference lives in its own world frame; align it to the DROID trajectory (rigid, Umeyama without scale)
ref_aligned = None
if ref is not None and len(ref_names) == len(names):
    A, B = ref[:, :3, 3], traj[:, :3, 3]
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    ref_aligned = (R @ (A - ma).T).T + mb
    ate = float(np.sqrt(np.mean(np.sum((ref_aligned - B) ** 2, axis=1))))
    print(f"ATE RMSE (rigid-aligned, vs reference trajectory): {ate:.3f} m")


def frustum(T, s=0.12):
    o = T[:3, 3]
    c = [o + T[:3, :3] @ np.array(v) * s for v in [(-.6, -.4, 1), (.6, -.4, 1), (.6, .4, 1), (-.6, .4, 1)]]
    pts = [o] + c
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    return pts, lines


def polyline(xyz, start_rgb=(0, .8, 0), end_rgb=(.9, 0, 0)):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(xyz),
                              o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(xyz) - 1)]))
    t = np.linspace(0, 1, max(len(xyz) - 1, 1))[:, None]
    ls.colors = o3d.utility.Vector3dVector(np.array(start_rgb) * (1 - t) + np.array(end_rgb) * t)
    return ls


def build_frustums():
    P, L = [], []
    for T in traj[:: args.frustum_every]:
        pts, lines = frustum(T)
        L += [[a + len(P), b + len(P)] for a, b in lines]
        P += pts
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(P), o3d.utility.Vector2iVector(L))
    ls.paint_uniform_color([0.1, 0.3, 0.9])
    return ls


# HTML export (map + trajectory + frustums). Points are base64 float32 so the file stays small.
xyz = np.asarray(pcd.points, dtype=np.float32)
rgb = np.asarray(pcd.colors, dtype=np.float32)
fr_pts, fr_lines = [], []
for T in traj[:: args.frustum_every]:
    pts, lines = frustum(T)
    fr_lines += [[a + len(fr_pts), b + len(fr_pts)] for a, b in lines]
    fr_pts += pts
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
data = {"xyz": b64(xyz), "rgb": b64(rgb), "traj": b64(traj[:, :3, 3]),
        "ref": b64(ref_aligned) if ref_aligned is not None else None,
        "fr": b64(np.array(fr_pts)), "fr_idx": np.array(fr_lines).ravel().tolist(), "n": int(len(xyz))}
html = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Map and trajectory</title>
<style>html,body{margin:0;height:100%;background:#15171a;color:#ddd;font:13px system-ui}
#ui{position:fixed;top:10px;left:10px;background:#0009;padding:8px 10px;border-radius:6px}label{display:block}</style>
<div id=ui><b>__N__ voxels</b><label><input type=checkbox id=t checked> trajectory</label>
<label><input type=checkbox id=f checked> camera frustums</label><label><input type=checkbox id=r checked> reference trajectory</label>
<span style="color:#7c7">green = start</span> <span style="color:#e66">red = end</span></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D=__DATA__;const f32=s=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer)};
const sc=new THREE.Scene();sc.background=new THREE.Color(0x15171a);
const cam=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,0.01,500);
const rd=new THREE.WebGLRenderer({antialias:true});rd.setSize(innerWidth,innerHeight);document.body.appendChild(rd.domElement);
const ctl=new THREE.OrbitControls(cam,rd.domElement);
// data is z-up; three.js is y-up, so rotate the whole group
const g=new THREE.Group();g.rotation.x=-Math.PI/2;sc.add(g);
let geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(f32(D.xyz),3));geo.setAttribute('color',new THREE.BufferAttribute(f32(D.rgb),3));
g.add(new THREE.Points(geo,new THREE.PointsMaterial({size:0.03,vertexColors:true})));
function line(arr,col){const p=f32(arr);const gg=new THREE.BufferGeometry();gg.setAttribute('position',new THREE.BufferAttribute(p,3));
 const cs=new Float32Array(p.length);const n=p.length/3;for(let i=0;i<n;i++){const t=i/(n-1||1);cs[3*i]=col?col[0]:t;cs[3*i+1]=col?col[1]:0.8*(1-t);cs[3*i+2]=col?col[2]:0;}
 gg.setAttribute('color',new THREE.BufferAttribute(cs,3));return new THREE.Line(gg,new THREE.LineBasicMaterial({vertexColors:true}));}
const tr=line(D.traj);g.add(tr);let rf=null;if(D.ref){rf=line(D.ref,[0.3,0.6,1]);g.add(rf);}
const fg=new THREE.BufferGeometry();fg.setAttribute('position',new THREE.BufferAttribute(f32(D.fr),3));fg.setIndex(D.fr_idx);
const fr=new THREE.LineSegments(fg,new THREE.LineBasicMaterial({color:0x4477ff}));g.add(fr);
g.add(new THREE.AxesHelper(0.5));
const p0=f32(D.traj);cam.position.set(p0[0]+3,2.5,-p0[1]+3);ctl.target.set(p0[0],0.5,-p0[1]);ctl.update();
t.onchange=e=>tr.visible=e.target.checked;f.onchange=e=>fr.visible=e.target.checked;r.onchange=e=>{if(rf)rf.visible=e.target.checked};
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rd.setSize(innerWidth,innerHeight)});
(function a(){requestAnimationFrame(a);ctl.update();rd.render(sc,cam)})();
</script>""".replace("__DATA__", json.dumps(data)).replace("__N__", str(data["n"]))
(root / "viewer.html").write_text(html)
print(f"HTML viewer: {root / 'viewer.html'}")
if args.html_only:
    raise SystemExit

geoms = {
    "map": pcd,
    "traj": polyline(traj[:, :3, 3]),
    "frustums": build_frustums(),
    "axes": o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
}
if ref_aligned is not None:
    geoms["ref"] = polyline(ref_aligned, (0.3, 0.6, 1), (0.3, 0.6, 1))
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("map + trajectory")
for g in geoms.values():
    vis.add_geometry(g)
shown = {k: True for k in geoms}


def toggler(key):
    def cb(v):
        if key not in geoms:
            return False
        (v.remove_geometry if shown[key] else v.add_geometry)(geoms[key], reset_bounding_box=False)
        shown[key] = not shown[key]
        return False
    return cb


vis.register_key_callback(ord("T"), toggler("traj"))
vis.register_key_callback(ord("F"), toggler("frustums"))
vis.register_key_callback(ord("Z"), toggler("ref"))
vis.run()
vis.destroy_window()
