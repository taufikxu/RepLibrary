# coding: utf-8
import time
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed

from Tools import FLAGS


class Trainer(object):
    def __init__(self, mOnline, mTarget, optim, ema_fun, augWrapper=None):
        self.mOnline = mOnline
        self.mTarget = mTarget
        self.optim = optim
        self.ema_fun = ema_fun
        self.augWrapper = augWrapper
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def energy(self, x, y=None, need_aug=True, require_grad_target=True):
        if need_aug is True:
            x = self.augWrapper(x)

        bs, ch = x.shape[0], x.shape[1] // 2
        x1, x2 = x[:, :ch], x[:, ch:]
        x_input = torch.cat([x1, x2], 0)

        _, pred, logit = self.mOnline(x_input)
        pred1, pred2 = pred[:bs], pred[bs:]
        logit1 = logit[:bs]

        if require_grad_target is True:
            projt, _, logit = self.mTarget(x_input)
            projt1, projt2 = projt[:bs], projt[bs:]
            logit2 = logit[:bs]
        else:
            with torch.no_grad():
                projt, _, logit = self.mTarget(x_input)
                projt1, projt2 = projt[:bs], projt[bs:]
                logit2 = logit[:bs]
        if y is not None:
            cla_loss = self.ce_loss(logit1, y) + self.ce_loss(logit2, y)
        else:
            cla_loss = 0

        repr_loss = torch.sum((pred1 - projt2) ** 2, 1) + torch.sum((pred2 - projt1) ** 2, 1)
        if y is None:
            return repr_loss
        else:
            return repr_loss, cla_loss

    def step(self):
        raise NotImplementedError()

    def sample(self, bs=None):
        raise NotImplementedError()
