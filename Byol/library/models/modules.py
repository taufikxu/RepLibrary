import time
import torch
import torch.nn as nn
from Tools import FLAGS

from library.data_iters import dataset_info, get_data_augmentation


def l2_norm(inputx):
    assert len(inputx.shape) == 2
    norm = torch.sqrt(torch.sum(inputx ** 2, 1)).view(-1, 1)
    return inputx / norm


def identity(x):
    return x


class mlpModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, norm_layer):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = norm_layer(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ByolWrapper(nn.Module):
    def __init__(self, backbone, projector, predictor, classifier, normalize="l2_norm"):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.predictor = predictor
        self.classifier = classifier
        self.normalize = eval(normalize)

    def forward(self, x, y=None):
        embedding = self.backbone(x)
        proj_out = self.projector(embedding)
        if y is not None:
            return embedding, self.normalize(proj_out)
        pred_out = self.predictor(proj_out)
        logit_cla = self.classifier(embedding.detach())

        return self.normalize(proj_out), self.normalize(pred_out), logit_cla

    def forward_cla(self, x):
        return self.classifier(x)


class EbmWrapper(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        energy, embedding = self.backbone(x, "1dim")
        logit_cla = self.classifier(embedding.detach())

        return energy, embedding, logit_cla

    def forward_cla(self, x):
        return self.classifier(x)
