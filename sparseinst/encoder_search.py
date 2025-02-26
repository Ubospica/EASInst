# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
# from detectron2.layers import Conv2d

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


class Conv2d(torch.nn.Conv2d):
    """
    This Conv2d supports mutable input channels
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"
        x = F.conv2d(
            x, self.weight[:, :x.shape[1]], self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


@SPARSE_INST_ENCODER_REGISTRY.register()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
        self.in_channels = [input_shape[f].channels for f in self.in_features]
        fpn_laterals = []
        fpn_outputs = []
        for in_channel in reversed(self.in_channels):
            lateral_conv = Conv2d(in_channel, self.num_channels, 1)
            output_conv = Conv2d(self.num_channels, self.num_channels, 3, padding=1)
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            fpn_laterals.append(lateral_conv)
            fpn_outputs.append(output_conv)
        self.fpn_laterals = nn.ModuleList(fpn_laterals)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        # ppm
        self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)
        # final fusion
        self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
        c2_msra_fill(self.fusion)

    def forward(self, features):
        features = [features[f] for f in self.in_features]
        features = features[::-1]
        prev_features = self.ppm(self.fpn_laterals[0](features[0]))
        outputs = [self.fpn_outputs[0](prev_features)]
        for feature, lat_conv, output_conv in zip(features[1:], self.fpn_laterals[1:], self.fpn_outputs[1:]):
            lat_features = lat_conv(feature)
            top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode='nearest')
            prev_features = lat_features + top_down_features
            outputs.insert(0, output_conv(prev_features))
        size = outputs[0].shape[2:]
        features = [
            outputs[0]] + [F.interpolate(x, size, mode='bilinear', align_corners=False) for x in outputs[1:]]
        features = self.fusion(torch.cat(features, dim=1))
        return features


def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)
