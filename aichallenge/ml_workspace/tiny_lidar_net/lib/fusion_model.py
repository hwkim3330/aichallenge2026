"""LiDAR + camera + speed, fused, predicting [accel, steer].

The AI track asks for an end-to-end model on 2D LiDAR, image and vehicle speed. What the
repo already had was TinyLidarNet on LiDAR alone (lib/model.py, input_dim 750, which
matches the 750 beams this simulator actually publishes on /sensing/lidar/scan), plus a
separate image pipeline in ml_workspace/pilot_net. This joins the three.

Why the speed input is not cosmetic
-----------------------------------
tiny_lidar_net/README.md records that training the accel head did not work and recommends
`loss.accel_weight=0.0`, i.e. steering only. That is what you would expect from the inputs:
a LiDAR frame says where the walls are but nothing about how fast the kart is going, and the
correct accelerator command at 8 m/s is the opposite of the correct one at 2 m/s in the same
place. The target is not a function of the input. Adding measured speed makes the accel head
learnable in principle; whether it learns is an experiment, not a claim.

Architecture
------------
    LiDAR  (B, 1, 750)   -> 1D CNN, the proven TinyLidarNet trunk, unchanged
    image  (B, 3, 66,200)-> small 2D CNN, PilotNet-shaped (the geometry that pipeline uses)
    speed  (B, 1)        -> 1 -> 16 MLP
                            concat -> 100 -> 50 -> 10 -> 2

The LiDAR trunk is kept identical to lib/model.py so the existing checkpoint
(ckpt/tinylidarnet_weights.npy, trained 2026-07-28 with a hairpin fine-tune) can initialise
it -- see `load_lidar_trunk`. Starting from that beats starting from scratch, and the data it
was trained on is gone (dataset/ is gitignored and empty), so it is not reproducible.

Deployment note: the scored environment runs numpy, not torch -- convert_weight.py exists
for exactly that. Keep every layer here to ops that script can emit.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LidarTrunk(nn.Module):
    """Byte-for-byte the conv stack of lib/model.py TinyLidarNet, so weights transfer."""

    def __init__(self, input_dim: int = 750):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        with torch.no_grad():
            d = torch.zeros(1, 1, input_dim)
            out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(d)))))
            self.out_dim = out.view(1, -1).shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x.flatten(1)


class ImageTrunk(nn.Module):
    """PilotNet-shaped 2D CNN. Expects (B, 3, 66, 200), the crop that pipeline uses."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 24, 5, stride=2)
        self.c2 = nn.Conv2d(24, 36, 5, stride=2)
        self.c3 = nn.Conv2d(36, 48, 5, stride=2)
        self.c4 = nn.Conv2d(48, 64, 3)
        self.c5 = nn.Conv2d(64, 64, 3)
        with torch.no_grad():
            d = torch.zeros(1, 3, 66, 200)
            out = self.c5(self.c4(self.c3(self.c2(self.c1(d)))))
            self.out_dim = out.view(1, -1).shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))
        return x.flatten(1)


class FusionNet(nn.Module):
    """LiDAR always; image and speed optional so the ablation is one flag, not one fork.

    Ablation matters here because the slide has to state what each input contributes, and
    "we added a camera" is not a finding. Same trunk, same data, `use_image=False` gives the
    comparison.
    """

    def __init__(self, lidar_dim: int = 750, output_dim: int = 2,
                 use_image: bool = True, use_speed: bool = True,
                 speed_embed: int = 16, dropout: float = 0.0):
        super().__init__()
        self.use_image = use_image
        self.use_speed = use_speed
        self.lidar = LidarTrunk(lidar_dim)
        feat = self.lidar.out_dim
        if use_image:
            self.image = ImageTrunk()
            feat += self.image.out_dim
        if use_speed:
            self.speed = nn.Sequential(nn.Linear(1, speed_embed), nn.ReLU())
            feat += speed_embed
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc1 = nn.Linear(feat, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, scan: Tensor, image: Optional[Tensor] = None,
                speed: Optional[Tensor] = None) -> Tensor:
        parts = [self.lidar(scan)]
        if self.use_image:
            if image is None:
                raise ValueError("use_image=True but no image given")
            parts.append(self.image(image))
        if self.use_speed:
            if speed is None:
                raise ValueError("use_speed=True but no speed given")
            parts.append(self.speed(speed.reshape(-1, 1)))
        x = self.drop(torch.cat(parts, dim=1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def load_lidar_trunk(model: FusionNet, npy_path: str, verbose: bool = True) -> int:
    """Warm-start the LiDAR trunk from the existing numpy checkpoint.

    ckpt/tinylidarnet_weights.npy is what convert_weight.py emits for deployment, so the
    key names follow that script rather than a torch state_dict. Only tensors whose shape
    matches are copied and the count is returned -- a silent partial load would be worse
    than no load, because the trunk would be half-random and the training curve would look
    like a modelling problem.
    """
    import numpy as np

    blob = np.load(npy_path, allow_pickle=True)
    weights = blob.item() if blob.dtype == object else dict(enumerate(blob))
    sd = model.lidar.state_dict()
    copied = 0
    for key, tensor in sd.items():
        for cand in (key, f"lidar.{key}", f"conv{key}", key.replace(".", "_")):
            if cand in weights:
                w = torch.as_tensor(np.asarray(weights[cand], dtype="float32"))
                if w.shape == tensor.shape:
                    sd[key] = w
                    copied += 1
                elif w.T.shape == tensor.shape:
                    sd[key] = w.T.contiguous()
                    copied += 1
                break
    model.lidar.load_state_dict(sd)
    if verbose:
        print(f"lidar trunk: {copied}/{len(sd)} 텐서 복원 "
              f"({'전체' if copied == len(sd) else '부분 — 키 이름 확인 필요'})")
    return copied
