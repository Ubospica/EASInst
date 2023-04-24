import argparse
import logging
import sys
from copy import deepcopy
import yaml

from sparseinst.search.common import Sequential
from sparseinst.search.genotype import print_recur

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from .models.common import *
from .models.experimental import *
from .models import darts_cell
from .utils.autoanchor import check_anchor_order
from .utils.general import make_divisible, check_file, set_logging
from .utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

def parse_model(d, ch=3):  # model_dict, input_channels(3)
    # logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # anchors, nc = d['anchors'], d['nc']
    gd, gw = 1, 1 #d['depth_multiple'], d['width_multiple']
    # na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    if not isinstance(ch, list):
        ch = [ch]
    layers, save, c2 = [], [], ch[-1] # layers, savelist, ch out
    arch_parameters = []
    op_arch_parameters = []
    ch_arch_parameters = []
    edge_arch_parameters = []
    search_space_per_layer = [] # for each layer that need to be searched, we construct a dict to record its candidate kernel sizes, dilation ratios and channel ratios
    d_copy = deepcopy(d)
    for i, (f, n, m, args) in enumerate(d):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # for j, a in enumerate(args):
        #     try:
        #         args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        #     except:
        #         pass
        if isinstance(args[-1], dict): args_dict = args[-1]; args = args[:-1]
        else: args_dict = {}

        id = None
        if 'id' in args_dict:
            id = args_dict['id']
            del args_dict['id']

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 加进来
        # 参数对应
        if m in [Conv, GhostConv, Bottleneck, Bottleneck_search, GhostBottleneck, SPP, DWConv, MixConv2d, Conv_search, Focus, CrossConv, BottleneckCSP, C3, C3_search, Conv_search_merge, Bottleneck_search_merge, C3_search_merge, SPP_search, Down_sampling_search_merge]:
            c1, c2 = ch[f], args[0]
            # if c2 != no:  # if not output
            c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3_search, C3_search_merge, Down_sampling_search_merge]:
                args.insert(2, n)  # number of repeats
                n = 1
        # 不一样的
        elif m in [Cells_search, Cells_search_merge, Cells]:
            if len(f) == 2:
              c_prev_prev, c_prev, c2 = ch[f[-2]], ch[f[-1]], args[2]
            else:
              c_prev_prev, c_prev, c2 = None, ch[f[-1]], args[2]
            # if c2 != no:  # if not output
            c2 = make_divisible(c2 * gw, 8)
            args[2] = c2
            args.insert(2, c_prev_prev)
            args.insert(3, c_prev)
            args_dict['N'] = n
            n = 1
        elif m in [AFF, FF]:
            c1s = [ch[x] for x in f]
            c2 = args[0]
            # if c2 != no:  # if not output
            c2 = make_divisible(c2 * gw, 8)
            args = [c1s, c2, *args[1:]]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args, **args_dict) for _ in range(n)]) if n > 1 else m(*args, **args_dict)  # module
        if m in [Conv_search, Bottleneck_search, C3_search, Conv_search_merge, Bottleneck_search_merge, C3_search_merge, Down_sampling_search_merge, AFF, Cells_search, Cells_search_merge]:
          m_list = [m_] if n==1 else m_
          for tmp in m_list:
            arch_parameters.extend(tmp.get_alphas())
            op_arch_parameters.extend(tmp.get_op_alphas())
            ch_arch_parameters.extend(tmp.get_ch_alphas())
            d = {'kd': args[1], 'ch_ratio': args[2]}
            search_space_per_layer.extend([d for _ in range(len(tmp.get_op_alphas()))])
        if m in [AFF]:
          edge_arch_parameters.extend(m_.get_edge_alphas())

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        m_.id = id
        m_.max_out_channels = c2

        logger.info('%3s%18s%3s%10.0f  %-40s%-30s%-30s' % (i, f, n, np, t, args, args_dict))  # print
        # save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        if m in [Conv_search_merge]:
          ch.append(int(c2*max(args[3])))
        elif m in [C3_search, Bottleneck_search, C3_search_merge, Down_sampling_search_merge, Bottleneck_search_merge]:
          ch.append(c2)
        elif m in [Cells_search, Cells_search_merge]:
          ch.append(c2*args[1])
        else: ch.append(c2)

    model = Sequential(*layers) # this is not nn.Sequential
    model.arch_parameters = arch_parameters
    model.op_arch_parameters = op_arch_parameters
    model.ch_arch_parameters = ch_arch_parameters
    model.edge_arch_parameters = edge_arch_parameters
    model.cfg = d_copy
    return model
