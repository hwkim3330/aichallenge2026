"""Check the two target conventions on a synthetic sequence before spending a training run on them.

What can silently go wrong here is the pair of mappings, not the plumbing. The dataset writes
2*v/v_max - 1 into slot 0 and the node inverts it with (head + 1)/2 * v_max, in two different
repositories' worth of code. If either side is off, training converges to something that drives at
the wrong speed and looks like a model problem.

So: build a sequence with known values, read it under both conventions, and assert the round trip.
"""
import os
import pathlib
import shutil
import sys
import tempfile

import numpy as np

ML = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML))

V_MAX = 8.33
N, PTS = 7, 750
SPEEDS = np.array([0.0, 1.0, 4.165, 8.33, 6.0, 2.5, 8.0], dtype=np.float32)
ACCELS = np.array([0.5, -0.2, 0.0, 1.0, -1.0, 0.3, 0.1], dtype=np.float32)
STEERS = np.array([0.0, 0.1, -0.1, 0.2, -0.2, 0.05, -0.05], dtype=np.float32)

tmp = pathlib.Path(tempfile.mkdtemp())
seq = tmp / "seq0"
seq.mkdir()
np.save(seq / "scans.npy", np.full((N, PTS), 15.0, dtype=np.float32))
np.save(seq / "steers.npy", STEERS)
np.save(seq / "accelerations.npy", ACCELS)
np.save(seq / "cmd_speeds.npy", SPEEDS)

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


def load():
    for m in [k for k in sys.modules if k.startswith("lib")]:
        del sys.modules[m]
    from lib.data import ScanControlSequenceDataset as SequenceDataset
    return SequenceDataset(str(seq), max_range=30.0)


print("default convention (acceleration in slot 0)")
os.environ.pop("TLN_SPEED_TARGET", None)
ds = load()
t = np.stack([ds[i][1] for i in range(N)])
check("slot 0 is the commanded acceleration", np.allclose(t[:, 0], ACCELS))
check("slot 1 is the steering", np.allclose(t[:, 1], STEERS))
check("scan normalised to [0,1]", 0.0 <= ds[0][0].min() and ds[0][0].max() <= 1.0,
      f"max {ds[0][0].max():.3f}")

print("\nspeed convention (normalised commanded speed in slot 0)")
os.environ["TLN_SPEED_TARGET"] = "1"
os.environ["TLN_SPEED_VMAX"] = str(V_MAX)
ds = load()
t = np.stack([ds[i][1] for i in range(N)])
expect = np.clip(2.0 * SPEEDS / V_MAX - 1.0, -1.0, 1.0)
check("slot 0 is the normalised speed", np.allclose(t[:, 0], expect, atol=1e-6),
      f"got {np.round(t[:, 0], 4).tolist()}")
check("slot 1 still the steering, order unchanged", np.allclose(t[:, 1], STEERS))
check("v=0 maps to -1", abs(t[0, 0] + 1.0) < 1e-6)
check("v=v_max maps to +1", abs(t[3, 0] - 1.0) < 1e-6)
check("v=v_max/2 maps to 0", abs(t[2, 0]) < 1e-3, f"got {t[2,0]:.6f}")

print("\nround trip through the node's inverse, (head + 1) / 2 * v_max")
recovered = (t[:, 0] + 1.0) * 0.5 * V_MAX
check("recovers the commanded speed", np.allclose(recovered, SPEEDS, atol=1e-4),
      f"max err {np.abs(recovered - SPEEDS).max():.2e}")

print("\nmissing cmd_speeds.npy must raise rather than silently fall back")
(seq / "cmd_speeds.npy").unlink()
try:
    load()
    check("raises FileNotFoundError", False, "it did not raise")
except FileNotFoundError as e:
    check("raises FileNotFoundError", "cmd_speeds.npy" in str(e), str(e)[:70])
except Exception as e:
    check("raises FileNotFoundError", False, f"raised {type(e).__name__} instead")

shutil.rmtree(tmp)
print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
