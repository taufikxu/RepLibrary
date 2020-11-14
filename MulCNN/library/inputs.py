import math
import torch
from torch import nn

from library import models
from library.data_iters import dataset_info
from Tools import FLAGS

augmentation_info = {"baseline": 3, "mulcnn": 12}


def get_model():
    device = FLAGS.device
    _, ncla, _ = dataset_info[FLAGS.dataset]
    kwargs = FLAGS.model.kwargs
    kwargs = vars(FLAGS.model.kwargs) if kwargs is not None else dict()
    kwargs.update({"input_channel": augmentation_info[FLAGS.augmentation]})
    module_object = models.get_models(FLAGS.model.name)
    kwargs["num_classes"] = ncla
    classifier = module_object(**kwargs).to(device)

    optim = get_optimizer(params=classifier.parameters(), **vars(FLAGS.optim))
    return classifier, optim


def get_optimizer(params, name, lr, beta1, beta2, weight_decay=0.0):
    if name.lower() == "adam":
        optim = torch.optim.Adam(params, lr, betas=(beta1, beta2))
    elif name.lower() == "nesterov":
        optim = torch.optim.SGD(params, lr, momentum=beta1, weight_decay=weight_decay, nesterov=True)
    elif name.lower() == "sgd":
        optim = torch.optim.SGD(params, lr, momentum=beta1, weight_decay=weight_decay, nesterov=False)
    else:
        raise ValueError("Unknown optimizer")

    return optim


def get_scheduler(optim):
    if FLAGS.training.scheduler.lower() == "warmup+cosine":
        _, _, nimages = dataset_info[FLAGS.dataset]
        n_iters_epoch = nimages // FLAGS.batch_size

        def lr_lambda(iters):
            warmup_epoch = 0.5
            assert iters < n_iters_epoch * (FLAGS.training.nepoch + 1)
            if iters < n_iters_epoch * warmup_epoch:
                return iters / (n_iters_epoch * warmup_epoch)
            real_iters = iters - n_iters_epoch * warmup_epoch
            total_iters = n_iters_epoch * (FLAGS.training.nepoch - warmup_epoch)
            relative = real_iters / total_iters * math.pi
            return 0.5 * (1 + math.cos(relative))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    else:
        raise ValueError("Unknown scheduler")

    return scheduler
