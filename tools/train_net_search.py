import os
import sys
import itertools
import time
from typing import Any, Dict, List, Set
import torch
from torch.utils.data import random_split

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetMapper
from detectron2.engine import AutogradProfiler, SimpleTrainer, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

sys.path.append(".")
from sparseinst import add_sparse_inst_config, COCOMaskEvaluator


from detectron2.data.build import (
    configurable, _train_loader_from_config, DatasetFromList, MapDataset, torchdata,
    TrainingSampler, build_batch_data_loader
)

@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader_splitted(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    ratio = .5
    dataset_train, dataset_val = random_split(dataset, [ratio, 1 - ratio])

    # if isinstance(dataset, torchdata.IterableDataset):
    #     assert sampler is None, "sampler must be None if dataset is IterableDataset"
    # else:
    #     if sampler is None:
    sampler_train = TrainingSampler(len(dataset_train))
    sampler_val = TrainingSampler(len(dataset_val))
        # assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset_train,
        sampler_train,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    ), build_batch_data_loader(
        dataset_val,
        sampler_val,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


from detectron2.engine.defaults import (
    logging, create_ddp_model, AMPTrainer, weakref
)
class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super(DefaultTrainer).__init__()
        # self._trainer = NASTrainer(self._trainer.model, self._trainer.data_loader, self._trainer.optimizer)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer_train = self.build_optimizer(cfg, model, True)
        optimizer_val = self.build_optimizer(cfg, model, False)
        data_loader_train, data_loader_val = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer_train = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader_train, optimizer_train
        )

        self._trainer_val = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader_val, optimizer_val
        )


        # self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOMaskEvaluator(dataset_name, ("segm", ), True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model, is_train=True):
        """
        is_train = True: train optimizer
                 = False: val optimizer
        """
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        if is_train:
            for key, value in model.named_parameters(recurse=True):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                if "backbone" in key:
                    lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                # for transformer
                if "patch_embed" in key or "cls_token" in key:
                    weight_decay = 0.0
                if "norm" in key:
                    weight_decay = 0.0
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        else:
            for value in model.arch_parameters:
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = cfg.SOLVER.BASE_LR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full  model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, amsgrad=cfg.SOLVER.AMSGRAD
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.SPARSE_INST.DATASET_MAPPER == "SparseInstDatasetMapper":
            from sparseinst import SparseInstDatasetMapper
            mapper = SparseInstDatasetMapper(cfg, is_train=True)
        else:
            mapper = None
        return build_detection_train_loader_splitted(cfg, mapper=mapper)

    def run_step(self):
        self._trainer_val.iter = self.iter
        self._trainer_val.run_step()
        self._trainer_train.iter = self.iter
        self._trainer_train.run_step()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "sparseinst" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sparseinst")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

    def genotype(self):
        op_geno=[]; op_geno_idx = []
        ch_geno = []; ch_geno_idx = []
        edge_geno = []; edge_geno_idx = []
        for alphas in self.op_arch_parameters:
            if alphas.dim() == 1:
              op_geno_idx.append(alphas.argmax(dim=-1).item())
            else:
              op_geno_idx.append(alphas)
        for alphas in self.ch_arch_parameters:
            if alphas is None: ch_geno_idx.append(None)
            else: ch_geno_idx.append(alphas.argmax(dim=-1).item())
        for alphas in self.edge_arch_parameters:
            edge_geno_idx.append([x.item() for x in torch.topk(alphas, k=2, dim=-1)[1]])
        # new yaml
        # eval: --cfg 导入cfg
        with open(self.cfg) as f:
            model_yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict
        idx_op = 0
        idx_ch = 0
        idx_edge = 0
        gd = model_yaml['depth_multiple']
        for i, tmp in enumerate(model_yaml['backbone'] + model_yaml['head']):
            if isinstance(tmp[3][-1], dict): # del unused variables
               for key in ['gumbel_channel']:
                 if key in tmp[3][-1].keys(): del tmp[3][-1][key]
            if tmp[2] in ['Conv_search', 'Bottleneck_search', 'Conv_search_merge', 'Bottleneck_search_merge', 'SepConv_search_merge']:
               n = tmp[1]
               n = max(round(n * gd), 1) if n > 1 else n  # depth gain
               func_p = tmp[3]
               Cout = func_p[0]
               tmp[2] = tmp[2].split('_')[0] # set name
               # set kernel-size and dilation-ratio and channel
               k = []; d = []; e = [];
               for j in range(n):
                 k.append(func_p[1][op_geno_idx[idx_op+j]][0])
                 d.append(func_p[1][op_geno_idx[idx_op+j]][1])
                 if ch_geno_idx[idx_ch+j] is None: e.append(1.0) # original YOLOv5 uses e=1.0
                 else: e.append(func_p[2][ch_geno_idx[idx_ch+j]])
               op_geno.append(list(zip(k,d)))
               ch_geno.append(e)
               if n == 1: k = k[0]; d=d[0]; e=e[0];
               tmp[3][1] = k
#               tmp[3].insert(2, d)
               tmp[3][2] = d # originally, tmp[3][2] is candidate_e, which is useless for full-train
               if tmp[2] in ['Bottleneck']:
                 if isinstance(tmp[3][-1], dict): tmp[3][-1]['e_bottleneck'] = e
                 else: tmp[3].append({'e_bottleneck':e})
               else: tmp[3][0] = Cout * e
               idx_op += n; idx_ch += n
            elif tmp[2] in ['C3_search', 'C3_search_merge']:
               n = tmp[1]
               n = max(round(n * gd), 1) if n > 1 else n  # depth gain
               func_p = tmp[3]
               Cout = func_p[0]
               candidate_e = func_p[2]
               tmp[2] = tmp[2].split('_')[0] # set name
               # set kernel-size and dilation-ratio and channel
               k = []; d = []; e = [];
               for j in range(n):
                 k.append(func_p[1][op_geno_idx[idx_op+j]][0])
                 d.append(func_p[1][op_geno_idx[idx_op+j]][1])
                 if ch_geno_idx[idx_ch+j] is None: e.append(1.0) # original YOLOv5 uses e=1.0
                 else: e.append(candidate_e[ch_geno_idx[idx_ch+j]])
               op_geno.append(list(zip(k,d)))
               ch_geno.append(deepcopy(e))
               if n == 1: k = k[0]; d=d[0]; e=e[0];
               tmp[3][1] = k
#               tmp[3].insert(2, d)
               tmp[3][2] = d
               if isinstance(tmp[3][-1], dict): tmp[3][-1]['e_bottleneck'] = e
               else: tmp[3].append({'e_bottleneck':e})
               # for c2
               if isinstance(func_p[-1], dict) and func_p[-1].get('search_c2', False):
                 if isinstance(func_p[-1]['search_c2'], list):
                   tmp[3][0] = Cout * func_p[-1]['search_c2'][ch_geno_idx[idx_ch+n]]
                   ch_geno[-1].append(func_p[-1]['search_c2'][ch_geno_idx[idx_ch+n]])
                 else:
                   tmp[3][0] = Cout * candidate_e[ch_geno_idx[idx_ch+n]]
                   ch_geno[-1].append(candidate_e[ch_geno_idx[idx_ch+n]])
                 del tmp[3][-1]['search_c2']
                 idx_ch += n+1
               else: idx_ch += n
               idx_op += n
            elif tmp[2] == 'AFF':
               tmp[2] = 'FF'
               all_edges = tmp[0]
               Cout, all_strides, all_kds = tmp[3][0:3]
               if isinstance(tmp[3][-1], dict):
                  candidate_e = tmp[3][-1].get('candidate_e', None)
                  separable = tmp[3][-1].get('separable', False)
               else: candidate_e = None; separable = False
               edges = []; ks= []; ds = []; strides = [];
               for j, idx in enumerate(edge_geno_idx[idx_edge]):
                  edges.append(all_edges[idx])
                  strides.append(all_strides[idx])
                  ks.append(all_kds[op_geno_idx[idx_op+idx]][0])
                  ds.append(all_kds[op_geno_idx[idx_op+idx]][1])
               edge_geno.append(edges)
               op_geno.append(list(zip(ks, ds)))
               ch_geno.append([1.0 for _ in range(len(edges))])
               # for Cout
               if ch_geno_idx[idx_ch+len(all_edges)] is not None:
                   Cout = Cout * candidate_e[ch_geno_idx[idx_ch+len(all_edges)]]
                   ch_geno[-1].append(candidate_e[ch_geno_idx[idx_ch+len(all_edges)]])
               args_dict = {'separable': separable}
               tmp[3] = [Cout, strides, ks, ds, args_dict]
               tmp[0] = edges
               idx_op += len(all_edges)
               idx_ch += len(all_edges)+1
               idx_edge += 1
            elif tmp[2] == 'SPP_search':
               tmp[2] = tmp[2].split('_')[0] # set name
            elif tmp[2] in ['Cells_search', 'Cells_search_merge']:
               tmp[2] = tmp[2].split('_')[0] # set name
               steps, multiplier, C, reduction, reduction_prev = tmp[3][0:5]
               op_alpha = op_geno_idx[idx_op]
               genotype, concat = darts_cell.genotype(op_alpha, steps, multiplier, num_input=len(tmp[0]))
               tmp[3] = [genotype, concat, C, reduction, reduction_prev]
               op_geno.append([genotype, concat])
               ch_geno.append([1])
               idx_op += 1
               idx_ch += 1
        assert(idx_ch == len(ch_geno_idx))
        assert(idx_op == len(op_geno_idx))
        assert(idx_edge == len(edge_geno_idx))
        geno = [op_geno, ch_geno, edge_geno] # split the alpha_op and alpha_channal
        model_yaml['geno'] = geno
        return geno, model_yaml

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


# def do_test(cfg, model):
#     results = OrderedDict()
#     for dataset_name in cfg.DATASETS.TEST:
#         data_loader = build_detection_test_loader(cfg, dataset_name)
#         evaluator = get_evaluator(
#             cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
#         )
#         results_i = inference_on_dataset(model, data_loader, evaluator)
#         results[dataset_name] = results_i
#         if comm.is_main_process():
#             logger.info("Evaluation results for {} in csv format:".format(dataset_name))
#             print_csv_format(results_i)
#     if len(results) == 1:
#         results = list(results.values())[0]
#     return results
