# coding: utf-8
import torch
import numpy as np

from Tools import FLAGS
from Tools.utils_torch import top_k_acc


class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.device = FLAGS.device
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, test_iters):
        self.model.eval()
        loss_list, acc_list = [], []
        with torch.no_grad():
            for x, y in test_iters:
                x, y = x.to(self.device), y.to(self.device)
                logit1 = self.model(x)
                loss_list.append(self.ce_loss(logit1, y).item())
                acc_list.append(top_k_acc(logit1, y, 1).item())

        self.model.train()

        return_dict = {"Loss": np.mean(loss_list), "Acc": np.mean(acc_list)}
        return return_dict
