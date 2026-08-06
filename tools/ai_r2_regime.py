"""Score a checkpoint by steering R^2 on held-out sequences, split by turn regime.

Loss is not the acceptance criterion. The training loss is a WeightedSmoothL1 dominated by the
straight-ahead frames, which are the majority at 20 Hz, so a model that predicts near zero everywhere
still reports a small loss. What decides whether the car completes a lap is whether it produces the
LARGE steering the corners need, so the metric is R^2 computed separately per |target| band, plus the
prediction standard deviation against the target's in each band.

Reported per band:
  n       frames in the band
  R2      1 - SSE/SST computed WITHIN the band, so it asks whether the model tracks variation there
          rather than being carried by the between-band spread
  std     prediction std vs target std -- an amplitude collapse shows here even when R^2 looks fine
  bias    mean(pred - target)
"""
import os
import pathlib
import sys

import numpy as np
import torch

B = pathlib.Path("/home/kim/aichallenge2026/aichallenge/ml_workspace/tiny_lidar_net")
sys.path.insert(0, str(B))
from lib.data import ScanControlSequenceDataset  # noqa: E402
from lib.model import TinyLidarNet  # noqa: E402

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else str(B / "checkpoints/exp341k/best_model.pth")
val_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else str(B / "dataset/exp_val"))

sd = torch.load(ckpt_path, map_location="cpu")
if isinstance(sd, dict) and "model_state_dict" in sd:
    print(f"checkpoint keys: {sorted(k for k in sd if k != 'model_state_dict')}")
    for k in ("epoch", "val_loss", "best_val_loss"):
        if k in sd:
            print(f"  {k}: {sd[k]}")
    sd = sd["model_state_dict"]

seqs = sorted(d for d in val_dir.iterdir() if d.is_dir() or d.is_symlink())

# input_dim must come from the DATA, not the model default. The default is 1080, which gives
# fc1 (100, 1792); this checkpoint's fc1 is (100, 1088), i.e. the 750-beam scan the simulator emits.
n_beams = int(np.load(seqs[0] / "scans.npy").shape[1])
model = TinyLidarNet(input_dim=n_beams)
assert model.fc1.weight.shape == sd["fc1.weight"].shape, (
    f"{n_beams} beams gives fc1 {tuple(model.fc1.weight.shape)}, "
    f"checkpoint has {tuple(sd['fc1.weight'].shape)}")
model.load_state_dict(sd)
model.eval()
print(f"input_dim {n_beams} -> fc1 {tuple(model.fc1.weight.shape)}")
P, T = [], []
per_seq = []
for d in seqs:
    ds = ScanControlSequenceDataset(d)
    # ds.scans is ALREADY clipped and divided by max_range in the constructor. Dividing again here
    # fed the model a 1/30 scale input and produced a near-constant +0.1 output with R^2 -0.27, which
    # looked like a broken model rather than a broken eval. Assert the range instead of trusting it.
    X = torch.from_numpy(ds.scans.astype(np.float32))
    assert X.max() <= 1.0 + 1e-6, f"{d.name}: scans not normalised, max {float(X.max()):.3f}"
    with torch.no_grad():
        out = model(X.unsqueeze(1)).numpy()
    p = out[:, 1]              # slot 1 is steering; slot 0 is the accel/speed head
    t = ds.steers.astype(np.float64)
    P.append(p)
    T.append(t)
    ss = 1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)
    per_seq.append((d.name, len(t), ss, float(np.std(p)), float(np.std(t))))

p = np.concatenate(P).astype(np.float64)
t = np.concatenate(T)
print(f"\ncheckpoint {ckpt_path}")
print(f"held-out: {len(seqs)} sequences, {len(t)} frames\n")


def band(lo, hi, label):
    m = (np.abs(t) >= lo) & (np.abs(t) < hi)
    if m.sum() < 20:
        print(f"  {label:<22} n={int(m.sum()):>7}  (too few to score)")
        return
    tt, pp = t[m], p[m]
    sst = ((tt - tt.mean()) ** 2).sum()
    r2 = 1 - ((pp - tt) ** 2).sum() / max(sst, 1e-12)
    print(f"  {label:<22} n={len(tt):>7}  R2={r2:+.3f}  "
          f"std pred {np.std(pp):.3f} / targ {np.std(tt):.3f}  bias {np.mean(pp - tt):+.4f}  "
          f"MAE {np.mean(np.abs(pp - tt)):.4f}")


print("overall")
band(0.0, 1e9, "all |steer|")
print("\nby regime (|target steering|, rad)")
band(0.00, 0.05, "near-straight <0.05")
band(0.05, 0.15, "gentle 0.05-0.15")
band(0.15, 0.30, "corner 0.15-0.30")
band(0.30, 1e9, "hard >0.30")

print("\nsign agreement on |target|>=0.10: "
      f"{np.mean(np.sign(p[np.abs(t) >= 0.10]) == np.sign(t[np.abs(t) >= 0.10])):.3f}")
print(f"saturation: |pred|>0.95*max {np.mean(np.abs(p) > 0.95 * np.abs(p).max()):.4f}, "
      f"pred range {p.min():+.3f}..{p.max():+.3f}, target range {t.min():+.3f}..{t.max():+.3f}")

print("\nper sequence (R2 over that sequence, std pred / targ)")
for name, n, ss, sp, st in per_seq:
    print(f"  {name:<44} n={n:>6}  R2={ss:+.3f}  {sp:.3f}/{st:.3f}")
