import torch
from Tools import FLAGS


class Trainer(object):
    def __init__(self, model, iters):
        self.model = model
        self.iters = iters
        self.device = FLAGS.device
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def step(self):
        x, y = self.iters.__next__()
        x, y = x.to(self.device), y.to(self.device)
        logit = self.model(x)
        cla_loss = self.ce_loss(logit, y)
        self.optim.zero_grad()
        cla_loss.backward()
        self.optim.step()

        return_dict = {"cla_loss": cla_loss.item()}
        return return_dict
