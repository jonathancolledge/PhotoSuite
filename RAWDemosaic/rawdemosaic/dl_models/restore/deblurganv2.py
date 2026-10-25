# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonathan Colledge
"""DeblurGANv2's FPNInception generator (VITA-Group/DeblurGANv2), reproduced
from the official models/fpn_inception.py, plus the InceptionResNetV2
backbone it wraps, from Cadene's pretrainedmodels/models/inceptionresnetv2.py
(BSD-3-Clause) -- verified tensor-shape-for-tensor-shape against the actual
"fpn_inception.h5" checkpoint (a plain torch.save'd state_dict despite the
.h5 extension -- a naming quirk in the upstream repo's own release, not an
actual HDF5/Keras file) before use, see restore/registry.py.

Vendored directly rather than added as a `pretrainedmodels` pip dependency:
that package is unmaintained, and its own `inceptionresnetv2()` factory
tries to download generic ImageNet weights on construction, which this
tool never needs -- our checkpoint already supplies every parameter
(including the backbone's) fully fine-tuned.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# InceptionResNetV2 backbone (Cadene/pretrained-models.pytorch, BSD-3-Clause)
# ---------------------------------------------------------------------------


class _BasicConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size, stride, padding: int = 0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class _Mixed5b(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _BasicConv2d(192, 96, 1, 1)
        self.branch1 = nn.Sequential(_BasicConv2d(192, 48, 1, 1), _BasicConv2d(48, 64, 5, 1, padding=2))
        self.branch2 = nn.Sequential(
            _BasicConv2d(192, 64, 1, 1), _BasicConv2d(64, 96, 3, 1, padding=1), _BasicConv2d(96, 96, 3, 1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), _BasicConv2d(192, 64, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), 1)


class _Block35(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.branch0 = _BasicConv2d(320, 32, 1, 1)
        self.branch1 = nn.Sequential(_BasicConv2d(320, 32, 1, 1), _BasicConv2d(32, 32, 3, 1, padding=1))
        self.branch2 = nn.Sequential(
            _BasicConv2d(320, 32, 1, 1), _BasicConv2d(32, 48, 3, 1, padding=1), _BasicConv2d(48, 64, 3, 1, padding=1)
        )
        self.conv2d = nn.Conv2d(128, 320, 1, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)
        out = self.conv2d(out)
        return self.relu(out * self.scale + x)


class _Mixed6a(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _BasicConv2d(320, 384, 3, 2)
        self.branch1 = nn.Sequential(
            _BasicConv2d(320, 256, 1, 1), _BasicConv2d(256, 256, 3, 1, padding=1), _BasicConv2d(256, 384, 3, 2)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x)), 1)


class _Block17(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.branch0 = _BasicConv2d(1088, 192, 1, 1)
        self.branch1 = nn.Sequential(
            _BasicConv2d(1088, 128, 1, 1),
            _BasicConv2d(128, 160, (1, 7), 1, padding=(0, 3)),
            _BasicConv2d(160, 192, (7, 1), 1, padding=(3, 0)),
        )
        self.conv2d = nn.Conv2d(384, 1088, 1, 1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch0(x), self.branch1(x)), 1)
        out = self.conv2d(out)
        return self.relu(out * self.scale + x)


class _Mixed7a(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(_BasicConv2d(1088, 256, 1, 1), _BasicConv2d(256, 384, 3, 2))
        self.branch1 = nn.Sequential(_BasicConv2d(1088, 256, 1, 1), _BasicConv2d(256, 288, 3, 2))
        self.branch2 = nn.Sequential(
            _BasicConv2d(1088, 256, 1, 1), _BasicConv2d(256, 288, 3, 1, padding=1), _BasicConv2d(288, 320, 3, 2)
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), 1)


class _Block8(nn.Module):
    def __init__(self, scale: float = 1.0, no_relu: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu
        self.branch0 = _BasicConv2d(2080, 192, 1, 1)
        self.branch1 = nn.Sequential(
            _BasicConv2d(2080, 192, 1, 1),
            _BasicConv2d(192, 224, (1, 3), 1, padding=(0, 1)),
            _BasicConv2d(224, 256, (3, 1), 1, padding=(1, 0)),
        )
        self.conv2d = nn.Conv2d(448, 2080, 1, 1)
        if not no_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat((self.branch0(x), self.branch1(x)), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        return out if self.no_relu else self.relu(out)


class InceptionResNetV2(nn.Module):
    """Full classification network (unused classifier head kept only
    because the FPN wrapper below saves/loads it via the same attribute
    path -- see FPN.__init__)."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.conv2d_1a = _BasicConv2d(3, 32, 3, 2)
        self.conv2d_2a = _BasicConv2d(32, 32, 3, 1)
        self.conv2d_2b = _BasicConv2d(32, 64, 3, 1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = _BasicConv2d(64, 80, 1, 1)
        self.conv2d_4a = _BasicConv2d(80, 192, 3, 1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = _Mixed5b()
        self.repeat = nn.Sequential(*[_Block35(scale=0.17) for _ in range(10)])
        self.mixed_6a = _Mixed6a()
        self.repeat_1 = nn.Sequential(*[_Block17(scale=0.10) for _ in range(20)])
        self.mixed_7a = _Mixed7a()
        self.repeat_2 = nn.Sequential(*[_Block8(scale=0.20) for _ in range(9)])
        self.block8 = _Block8(no_relu=True)
        self.conv2d_7b = _BasicConv2d(2080, 1536, 1, 1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)
        self.last_linear = nn.Linear(1536, num_classes)


# ---------------------------------------------------------------------------
# FPN + FPNInception generator (VITA-Group/DeblurGANv2, BSD/CC-BY-NC per repo)
# ---------------------------------------------------------------------------


class FPN(nn.Module):
    def __init__(self, norm_layer, num_filters: int = 256, inception_num_classes: int = 1000) -> None:
        super().__init__()
        self.inception = InceptionResNetV2(num_classes=inception_num_classes)

        self.enc0 = self.inception.conv2d_1a
        self.enc1 = nn.Sequential(self.inception.conv2d_2a, self.inception.conv2d_2b, self.inception.maxpool_3a)
        self.enc2 = nn.Sequential(self.inception.conv2d_3b, self.inception.conv2d_4a, self.inception.maxpool_5a)
        self.enc3 = nn.Sequential(self.inception.mixed_5b, self.inception.repeat, self.inception.mixed_6a)
        self.enc4 = nn.Sequential(self.inception.repeat_1, self.inception.mixed_7a)

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU(inplace=True))
        self.pad = nn.ReflectionPad2d(1)
        self.lateral4 = nn.Conv2d(2080, num_filters, 1, bias=False)
        self.lateral3 = nn.Conv2d(1088, num_filters, 1, bias=False)
        self.lateral2 = nn.Conv2d(192, num_filters, 1, bias=False)
        self.lateral1 = nn.Conv2d(64, num_filters, 1, bias=False)
        self.lateral0 = nn.Conv2d(32, num_filters // 2, 1, bias=False)

    def forward(self, x: torch.Tensor):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        lateral4 = self.pad(self.lateral4(enc4))
        lateral3 = self.pad(self.lateral3(enc3))
        lateral2 = self.lateral2(enc2)
        lateral1 = self.pad(self.lateral1(enc1))
        lateral0 = self.lateral0(enc0)

        pad = (1, 2, 1, 2)
        pad1 = (0, 1, 0, 1)
        map4 = lateral4
        map3 = self.td1(lateral3 + F.interpolate(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(F.pad(lateral2, pad, "reflect") + F.interpolate(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + F.interpolate(map2, scale_factor=2, mode="nearest"))
        return F.pad(lateral0, pad1, "reflect"), map1, map2, map3, map4


class FPNHead(nn.Module):
    def __init__(self, num_in: int, num_mid: int, num_out: int) -> None:
        super().__init__()
        self.block0 = nn.Conv2d(num_in, num_mid, 3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.block0(x), inplace=True)
        return F.relu(self.block1(x), inplace=True)


class FPNInception(nn.Module):
    def __init__(
        self,
        norm_layer,
        output_ch: int = 3,
        num_filters: int = 128,
        num_filters_fpn: int = 256,
        inception_num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer, inception_num_classes=inception_num_classes)
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.smooth = nn.Sequential(nn.Conv2d(4 * num_filters, num_filters, 3, padding=1), norm_layer(num_filters), nn.ReLU())
        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, 3, padding=1), norm_layer(num_filters // 2), nn.ReLU()
        )
        self.final = nn.Conv2d(num_filters // 2, output_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = F.interpolate(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = F.interpolate(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = F.interpolate(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = F.interpolate(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = F.interpolate(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = F.interpolate(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        res = torch.tanh(final) + x
        return torch.clamp(res, min=-1, max=1)


def is_deblurganv2_state_dict(state_dict: dict) -> bool:
    return any(k.startswith("fpn.inception.conv2d_1a.") for k in state_dict) or any(
        k.startswith("module.fpn.inception.conv2d_1a.") for k in state_dict
    )


def build_deblurganv2(state_dict: dict) -> tuple[nn.Module, dict]:
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    output_ch = state_dict["final.weight"].shape[0]
    num_filters = state_dict["smooth2.0.weight"].shape[0] * 2
    num_filters_fpn = state_dict["fpn.lateral4.weight"].shape[0]
    inception_num_classes = state_dict["fpn.inception.last_linear.weight"].shape[0]
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)

    model = FPNInception(
        norm_layer=norm_layer, output_ch=output_ch, num_filters=num_filters,
        num_filters_fpn=num_filters_fpn, inception_num_classes=inception_num_classes,
    )
    model.load_state_dict(state_dict, strict=True)
    meta = {
        "architecture": "DeblurGANv2-FPNInception", "in_channels": 3, "out_channels": output_ch,
        "size_multiple": 32,  # 5 stride-2 stages in the Inception backbone, no internal padding
        # Empirically: 96px fails (a BatchNorm layer, now in train mode --
        # see inference_mode below -- hits a 1x1 spatial map and refuses to
        # compute batch statistics from a single value), 128px works. The
        # Inception stem's unpadded ("valid") convolutions crop several
        # pixels at nearly every layer, so being merely a multiple of 32
        # isn't sufficient; a pathological edge tile needs an explicit floor.
        "min_tile_size": 160,
        # The official predict.py runs the generator via model.train(True), not
        # .eval(), with the comment "GAN inference should be in train mode to
        # use actual stats in norm layers, it's not a bug": its InstanceNorm2d
        # layers use track_running_stats=True, so .eval() would use frozen
        # running stats instead of this image's own (which InstanceNorm exists
        # to compute) -- producing systematically wrong per-channel scaling.
        "inference_mode": "train",
    }
    return model, meta
