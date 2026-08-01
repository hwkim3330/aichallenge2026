import sys, numpy as np, torch
sys.path.insert(0, '/aichallenge/ml_workspace/tiny_lidar_net')
from lib.model import TinyLidarNet
from lib.data import MultiSeqConcatDataset
from torch.utils.data import DataLoader

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = TinyLidarNet(input_dim=750, output_dim=2).to(dev)
m.load_state_dict(torch.load('/aichallenge/ml_workspace/tiny_lidar_net/checkpoints/baseline_20260802/best_model.pth', map_location=dev))
m.eval()
ds = MultiSeqConcatDataset('/aichallenge/ml_workspace/tiny_lidar_net/dataset/val')
dl = DataLoader(ds, batch_size=256, shuffle=False)
P, T = [], []
with torch.no_grad():
    for s, t in dl:
        P.append(m(s.unsqueeze(1).to(dev)).cpu().numpy()); T.append(t.numpy())
P = np.concatenate(P); T = np.concatenate(T)

def sl1(d):
    d = np.abs(d); return np.where(d < 1.0, 0.5*d*d, d-0.5).mean()

for i, name in enumerate(['accel', 'steer']):
    print(f"{name}: SmoothL1={sl1(P[:,i]-T[:,i]):.4f}  MAE={np.abs(P[:,i]-T[:,i]).mean():.4f}")
    print(f"    pred [{P[:,i].min():+.3f},{P[:,i].max():+.3f}] mean={P[:,i].mean():+.3f} std={P[:,i].std():.4f}")
    print(f"    targ [{T[:,i].min():+.3f},{T[:,i].max():+.3f}] mean={T[:,i].mean():+.3f} std={T[:,i].std():.4f}")
print(f"total (sum of channel means) = {sl1(P[:,0]-T[:,0])+sl1(P[:,1]-T[:,1]):.4f}")
# best possible accel given tanh
best = np.clip(T[:,0], -1, 1)
print(f"accel floor if model emitted the clipped target exactly: {sl1(best-T[:,0]):.4f}")
# steer baseline: predict the mean
print(f"steer loss if model always predicted the target mean: {sl1(T[:,1].mean()-T[:,1]):.4f}")
