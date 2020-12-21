import math
import torch
from torch import nn

from library import models
from library.data_iters import dataset_info
from Tools import FLAGS

from torchlars import LARS as torchlars_LARS


class OptimizerWrapper(object):
    def __init__(self, opt_list):
        self.opt_list = opt_list

    def zero_grad(self):
        for opt in self.opt_list:
            opt.zero_grad()

    def step(self):
        for opt in self.opt_list:
            opt.step()


class SchedulerWrapper(object):
    def __init__(self, sch_list):
        self.sch_list = sch_list

    def step(self):
        for sch in self.sch_list:
            sch.step()


def add_weight_decay(model, weight_decay=0.0000015, skip_list=("bn", "bias")):
    decay = []
    no_decay = []
    no_decay_name = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        for to_filter in skip_list:
            if to_filter in name and name not in no_decay_name:
                no_decay.append(param)
                no_decay_name.add(name)
        if name not in no_decay_name:
            decay.append(param)

    assert len(no_decay) == len(no_decay_name)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def get_model():
    device = FLAGS.device
    _, ncla, _ = dataset_info[FLAGS.dataset]
    kwargs = FLAGS.model.backbone_kwargs
    kwargs = vars(FLAGS.model.backbone_kwargs) if kwargs is not None else dict()
    module_object, backbone_dim = models.get_models(FLAGS.model.backbone_name)
    classifier = nn.Linear(backbone_dim, ncla).to(device)
    if FLAGS.method == "ebm":
        backbone = module_object(**kwargs).to(device)
        model = models.EbmWrapper(backbone, classifier)
    else:
        kwargs["norm_layer"] = models.get_norm_layer(FLAGS.model.backbone_norm)
        backbone = module_object(**kwargs).to(device)

        head_norm = models.get_norm_layer(FLAGS.model.head_norm)
        projector = models.mlpModule(
            backbone_dim, FLAGS.model.projector.hidden_size, FLAGS.model.projector.output_size, norm_layer=head_norm,
        ).to(device)

        predictor = models.mlpModule(
            FLAGS.model.projector.output_size,
            FLAGS.model.predictor.hidden_size,
            FLAGS.model.predictor.output_size,
            norm_layer=head_norm,
        ).to(device)

        assert FLAGS.model.predictor.output_size == FLAGS.model.projector.output_size
        model = models.ByolWrapper(backbone, projector, predictor, classifier, normalize=FLAGS.model.feature_norm)

    optim = get_optimizer(model.parameters(), **vars(FLAGS.optim))
    return model, optim


def get_optimizer(params, name, lr, beta1, beta2, weight_decay=0.0):
    if name.lower() == "adam":
        optim = torch.optim.Adam(params, lr, betas=(beta1, beta2))
    elif name.lower() == "nesterov":
        optim = torch.optim.SGD(params, lr, momentum=beta1, weight_decay=weight_decay, nesterov=True)
    elif name.lower() == "sgd":
        optim = torch.optim.SGD(params, lr, momentum=beta1, weight_decay=weight_decay, nesterov=False)
    elif name.lower() == "lars":
        optim = torch.optim.SGD(params, lr, momentum=beta1, weight_decay=weight_decay)
        optim = torchlars_LARS(optim)
    else:
        raise ValueError("Unknown optimizer")
    return optim


def get_scheduler(optim):
    if FLAGS.training.scheduler.lower() == "byol":
        _, _, nimages = dataset_info[FLAGS.dataset]
        n_iters_epoch = nimages // FLAGS.batch_size

        def lr_lambda(iters):
            assert iters < n_iters_epoch * (FLAGS.training.nepoch + 1)
            if iters < n_iters_epoch * 10:
                return iters / (n_iters_epoch * 10)
            real_iters = iters - n_iters_epoch * 10
            total_iters = n_iters_epoch * (FLAGS.training.nepoch - 10)
            relative = real_iters / total_iters * math.pi
            return 0.5 * (1 + math.cos(relative))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    else:
        raise ValueError("Unknown scheduler")

    return scheduler


def get_ema_coefficiency():
    if FLAGS.training.ema_coe.lower() == "byol":
        _, _, nimages = dataset_info[FLAGS.dataset]
        n_iters_epoch = nimages // FLAGS.batch_size

        def ema_func(iters):
            total_iters = n_iters_epoch * FLAGS.training.nepoch
            relative = iters / total_iters * math.pi
            convex_coe = 0.5 * (1 + math.cos(relative))
            return FLAGS.training.ema_coe_val * convex_coe + 1.0 * (1 - convex_coe)

    else:
        raise ValueError("Unknown ema scheduler")

    return ema_func
