# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np

from Tools import FLAGS
from Tools.utils_torch import update_average, top_k_acc


class Evaluator(object):
    def __init__(self, mOnline, mTarget):
        self.mOnline = mOnline
        self.mTarget = mTarget
        self.device = FLAGS.device
        self.ce_loss = torch.nn.CrossEntropyLoss()
        # self.mean = [0.4914, 0.4822, 0.4465]
        # self.std = [0.2470, 0.2435, 0.2616]

    def __call__(self, test_iters):
        accOnline, accTarget = [], []
        lossOnline, lossTarget = [], []
        self.mOnline.eval()
        self.mTarget.eval()
        with torch.no_grad():
            for x, y in test_iters:
                x, y = x.to(self.device), y.to(self.device)
                _, _, logit1 = self.mOnline(x)
                lossOnline.append(self.ce_loss(logit1, y).item())
                accOnline.append(top_k_acc(logit1, y, 1).item())

                _, _, logit1 = self.mTarget(x)
                lossTarget.append(self.ce_loss(logit1, y).item())
                accTarget.append(top_k_acc(logit1, y, 1).item())
        self.mOnline.train()
        self.mTarget.train()

        return_dict = {
            "LossOnline": np.mean(lossOnline),
            "LossTarget": np.mean(lossTarget),
            "accOnline": np.mean(accOnline),
            "accTarget": np.mean(accTarget),
        }
        return return_dict


class KNN_evaluator(object):
    def __init__(self, mOnline, mTarget):
        self.mOnline = mOnline
        self.mTarget = mTarget
        self.device = FLAGS.device
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, train_iters, test_iters, K=200):
        trainsize = train_iters.dataset.__len__()
        testsize = test_iters.dataset.__len__()
        trainFeatures = np.zeros((FLAGS.model.projector.output_size, trainsize))
        trainLabels = np.zeros((trainsize))

        self.mOnline.eval()
        self.mTarget.eval()

        total = 0

        with torch.no_grad():
            for batch_idx, (x_aug, y) in enumerate(train_iters):
                # accomodate two crop for MoCO
                # inputs = inputs[0]
                batchSize = x.size(0)

                x_aug, y = x_aug.to(self.device), y.to(self.device)
                x = x_aug[:, :3, :, :]
                features, _, _ = self.mTarget(x)
                trainFeatures[:, batch_idx * batchSize : batch_idx * batchSize + batchSize] = (
                    features.data.cpu().numpy().T
                )  # features.data.t() # Peter TODO: torch0.4?
                trainLabels[batch_idx * batchSize : batch_idx * batchSize + batchSize] = y.data.cpu()

        trainLabels = torch.LongTensor(trainLabels).cuda()
        C = trainLabels.max() + 1
        C = np.int(C)

        trainFeatures = torch.Tensor(trainFeatures).cuda()
        top1 = 0.0
        top5 = 0.0

        with torch.no_grad():
            retrieval_one_hot = torch.zeros(K, C).cuda()
            for batch_idx, (x_aug, y) in enumerate(test_iters):

                x, y = x_aug.to(self.device), y.to(self.device)
                features, _, _ = self.mTarget(x)

                # inputs = inputs[0]
                # targets = targets.cuda()
                # batchSize = inputs.size(0)
                batchSize = x.size(0)

                # model.cuda()
                # inputs.cuda()
                # features = model(inputs)
                total += y.size(0)

                dist = torch.mm(features, trainFeatures)
                yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
                candidates = trainLabels.view(1, -1).expand(batchSize, -1)
                retrieval = torch.gather(candidates, 1, yi)

                retrieval_one_hot.resize_(batchSize * K, C).zero_()
                retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
                yd_transform = yd.clone().div_(1.0).exp_()
                probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1),), 1,)
                _, predictions = probs.sort(1, True)

                # Find which predictions match the target
                correct = predictions.eq(y.data.view(-1, 1))

                top1 = top1 + correct.narrow(1, 0, 1).sum().item()
                top5 = top5 + correct.narrow(1, 0, 5).sum().item()
                # print(3.24, batch_idx)

        self.mOnline.train()
        self.mTarget.train()

        print(top1 * 100.0 / total)
        return_dict = {
            # "accOnline": np.mean(accOnline),
            "accKNN_top1": top1 * 100.0 / total,
            "accKNN_top5": top5 * 100.0 / total,
        }
        return return_dict

