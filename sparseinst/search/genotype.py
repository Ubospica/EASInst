
import torch
import yaml
from .models import darts_cell
from copy import deepcopy

def print_recur(val):
    print("print::", type(val), val)
    if isinstance(val, str):
        return
    try:
        for x in val:
            print_recur(x)
    except TypeError:
        pass


def genotype(model):
    cfg = model.cfg
    op_geno=[]; op_geno_idx = []
    ch_geno = []; ch_geno_idx = []
    edge_geno = []; edge_geno_idx = []
    for alphas in model.op_arch_parameters:
        if alphas.dim() == 1:
            op_geno_idx.append(alphas.argmax(dim=-1).item())
        else:
            op_geno_idx.append(alphas)
    for alphas in model.ch_arch_parameters:
        if alphas is None: ch_geno_idx.append(None)
        else: ch_geno_idx.append(alphas.argmax(dim=-1).item())
    for alphas in model.edge_arch_parameters:
        edge_geno_idx.append([x.item() for x in torch.topk(alphas, k=2, dim=-1)[1]])
    # new yaml
    # eval: --cfg 导入cfg
    # with open(self.cfg) as f:
        # model_yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
    idx_op = 0
    idx_ch = 0
    idx_edge = 0
    # gd = model_yaml['depth_multiple']
    gd = 1
    for tmp in cfg:
        if isinstance(tmp[3][-1], dict):  # del unused variables
            for key in ["gumbel_channel"]:
                if key in tmp[3][-1].keys():
                    del tmp[3][-1][key]
        if tmp[2] in [
            "Conv_search",
            "Bottleneck_search",
            "Conv_search_merge",
            "Bottleneck_search_merge",
            "SepConv_search_merge",
        ]:
            n = tmp[1]
            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            func_p = tmp[3]
            Cout = func_p[0]
            tmp[2] = tmp[2].split("_")[0]  # set name
            # set kernel-size and dilation-ratio and channel
            k = []
            d = []
            e = []
            for j in range(n):
                k.append(func_p[1][op_geno_idx[idx_op + j]][0])
                d.append(func_p[1][op_geno_idx[idx_op + j]][1])
                if ch_geno_idx[idx_ch + j] is None:
                    e.append(1.0)  # original YOLOv5 uses e=1.0
                else:
                    e.append(func_p[2][ch_geno_idx[idx_ch + j]])
            op_geno.append(list(zip(k, d)))
            ch_geno.append(e)
            if n == 1:
                k = k[0]
                d = d[0]
                e = e[0]
            tmp[3][1] = k
            #               tmp[3].insert(2, d)
            tmp[3][
                2
            ] = d  # originally, tmp[3][2] is candidate_e, which is useless for full-train
            if tmp[2] in ["Bottleneck"]:
                if isinstance(tmp[3][-1], dict):
                    tmp[3][-1]["e_bottleneck"] = e
                else:
                    tmp[3].append({"e_bottleneck": e})
            else:
                tmp[3][0] = Cout * e
            idx_op += n
            idx_ch += n
        elif tmp[2] in ["C3_search", "C3_search_merge", "Down_sampling_search_merge"]:
            n = tmp[1]
            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            func_p = tmp[3]
            Cout = func_p[0]
            candidate_e = func_p[2]
            tmp[2] = tmp[2].split("_")[0]  # set name
            # set kernel-size and dilation-ratio and channel
            k = []
            d = []
            e = []
            for j in range(n):
                k.append(func_p[1][op_geno_idx[idx_op + j]][0])
                d.append(func_p[1][op_geno_idx[idx_op + j]][1])
                if ch_geno_idx[idx_ch + j] is None:
                    e.append(1.0)  # original YOLOv5 uses e=1.0
                else:
                    e.append(candidate_e[ch_geno_idx[idx_ch + j]])
            op_geno.append(list(zip(k, d)))
            ch_geno.append(deepcopy(e))
            if n == 1:
                k = k[0]
                d = d[0]
                e = e[0]
            tmp[3][1] = k
            #               tmp[3].insert(2, d)
            tmp[3][2] = d
            if isinstance(tmp[3][-1], dict):
                tmp[3][-1]["e_bottleneck"] = e
            else:
                tmp[3].append({"e_bottleneck": e})
            # for c2
            if isinstance(func_p[-1], dict) and func_p[-1].get("search_c2", False):
                if isinstance(func_p[-1]["search_c2"], list):
                    tmp[3][0] = (
                        Cout * func_p[-1]["search_c2"][ch_geno_idx[idx_ch + n]]
                    )
                    ch_geno[-1].append(
                        func_p[-1]["search_c2"][ch_geno_idx[idx_ch + n]]
                    )
                else:
                    tmp[3][0] = Cout * candidate_e[ch_geno_idx[idx_ch + n]]
                    ch_geno[-1].append(candidate_e[ch_geno_idx[idx_ch + n]])
                del tmp[3][-1]["search_c2"]
                idx_ch += n + 1
            else:
                idx_ch += n
            idx_op += n
        elif tmp[2] == "AFF":
            tmp[2] = "FF"
            all_edges = tmp[0]
            Cout, all_strides, all_kds = tmp[3][0:3]
            if isinstance(tmp[3][-1], dict):
                candidate_e = tmp[3][-1].get("candidate_e", None)
                separable = tmp[3][-1].get("separable", False)
            else:
                candidate_e = None
                separable = False
            edges = []
            ks = []
            ds = []
            strides = []
            for j, idx in enumerate(edge_geno_idx[idx_edge]):
                edges.append(all_edges[idx])
                strides.append(all_strides[idx])
                ks.append(all_kds[op_geno_idx[idx_op + idx]][0])
                ds.append(all_kds[op_geno_idx[idx_op + idx]][1])
            edge_geno.append(edges)
            op_geno.append(list(zip(ks, ds)))
            ch_geno.append([1.0 for _ in range(len(edges))])
            # for Cout
            if ch_geno_idx[idx_ch + len(all_edges)] is not None:
                Cout = Cout * candidate_e[ch_geno_idx[idx_ch + len(all_edges)]]
                ch_geno[-1].append(
                    candidate_e[ch_geno_idx[idx_ch + len(all_edges)]]
                )
            args_dict = {"separable": separable}
            tmp[3] = [Cout, strides, ks, ds, args_dict]
            tmp[0] = edges
            idx_op += len(all_edges)
            idx_ch += len(all_edges) + 1
            idx_edge += 1
        elif tmp[2] == "SPP_search":
            tmp[2] = tmp[2].split("_")[0]  # set name
        elif tmp[2] in ["Cells_search", "Cells_search_merge"]:
            tmp[2] = tmp[2].split("_")[0]  # set name
            steps, multiplier, C, reduction, reduction_prev = tmp[3][0:5]
            op_alpha = op_geno_idx[idx_op]
            genotype, concat = darts_cell.genotype(
                op_alpha, steps, multiplier, num_input=len(tmp[0])
            )
            tmp[3] = [genotype, concat, C, reduction, reduction_prev]
            op_geno.append([genotype, concat])
            ch_geno.append([1])
            idx_op += 1
            idx_ch += 1
    assert idx_ch == len(ch_geno_idx)
    assert idx_op == len(op_geno_idx)
    assert idx_edge == len(edge_geno_idx)
    geno = [op_geno, ch_geno, edge_geno]  # split the alpha_op and alpha_channal
    # model_yaml["geno"] = geno
    return geno, cfg
