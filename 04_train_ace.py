"""Train DovSG's vendored ACE on ~90 percent of the frames; the rest are held out for 05_eval_reloc.py.

Run in the DovSG env (needs the ACE C++ extension dsacstar and a GPU). DovSG is only read, not modified.
"""
import os
import sys
import shutil
from pathlib import Path

import config
from common import scene_args, scene_dir, frame_names

p = scene_args(__doc__)
p.add_argument("--holdout_every", type=int, default=config.HOLDOUT_EVERY)
args = p.parse_args()

root = scene_dir(args.scene)
names = [n for n in frame_names(root) if (root / "poses" / f"{n}.txt").exists()]
held = {n for i, n in enumerate(names) if i % args.holdout_every == args.holdout_every // 2}
train = [n for n in names if n not in held]

tr = root / "ace_train"
if tr.exists():
    shutil.rmtree(tr)
for sub, ext in [("rgb", "jpg"), ("poses", "txt"), ("calibration", "txt")]:
    (tr / sub).mkdir(parents=True)
    for n in train:
        os.symlink((root / sub / f"{n}.{ext}").resolve(), tr / sub / f"{n}.{ext}")
(root / "heldout.txt").write_text("\n".join(sorted(held)))
print(f"train {len(train)} frames, held out {len(held)}")

out = (root / "ace" / "ace.pt").resolve()
out.parent.mkdir(parents=True, exist_ok=True)

os.chdir(config.DOVSG_ROOT)             # ACE reads its configs with relative paths
sys.path.insert(0, str(config.DOVSG_ROOT))

# Newer PyTorch no longer sets optimizer._step_count, which ACE's trainer reads to detect skipped AMP steps.
# Count real optimizer steps ourselves instead of editing the DovSG/ACE sources.
import torch.optim as optim
if not hasattr(optim.AdamW(__import__("torch").nn.Linear(1, 1).parameters()), "_step_count"):
    _AdamW = optim.AdamW

    class CountingAdamW(_AdamW):
        _step_count = 0

        def step(self, *a, **k):
            out = super().step(*a, **k)
            self._step_count += 1
            return out

    optim.AdamW = CountingAdamW

from ace.train_ace import train_ace
train_ace(tr.resolve(), out)
print(f"ACE head saved to {out}")
