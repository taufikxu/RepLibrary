# coding: utf-8
import time
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd

from Tools import FLAGS
from Tools.utils_torch import update_average
from library.data_iters import dataset_info
from library.trainer import trainer_base


class Trainer(trainer_base.Trainer):
    def __init__(self, mOnline, mTarget, optim, ema_fun, augWrapper=None):
        super().__init__(mOnline, mTarget, optim, ema_fun, augWrapper)

    def step(self, x_real, y_real, iters):
        x_real = x_real.detach()
        repr_loss, cla_loss = self.energy(x_real, y_real, need_aug=False, require_grad_target=False)
        repr_loss = repr_loss.mean(dim=0)

        loss = cla_loss + repr_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        update_average(self.mTarget, self.mOnline, self.ema_fun(iters))

        return_dict = {
            "cla_loss": cla_loss.item(),
            "repr_loss": repr_loss.item(),
            "ema_coe_val": self.ema_fun(iters),
            "lr_online": self.optim.opt_list[0].param_groups[0]["lr"],
            "lr_target": self.optim.opt_list[1].param_groups[0]["lr"],
        }

        return return_dict

    def sample(self, bs=None):
        bs = FLAGS.batch_size if bs is None else bs
        img_size = dataset_info[FLAGS.dataset][0]
        return torch.rand((bs, 3, img_size, img_size))
