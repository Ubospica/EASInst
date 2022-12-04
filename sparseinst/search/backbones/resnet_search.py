
import math
import torch.nn as nn
from timm.models.resnet import BasicBlock, Bottleneck
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame

from detectron2.layers import ShapeSpec, FrozenBatchNorm2d
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import NaiveSyncBatchNorm, DeformConv

from sparseinst.search.parse_model import parse_model


@BACKBONE_REGISTRY.register()
def build_search_backbone(cfg, input_shape):

    backbone_cfg = cfg.MODEL.BACKBONE.ARGUMENTS

    return parse_model(backbone_cfg)
