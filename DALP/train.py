import torch
import numpy as np
from collections import defaultdict

from Tools.logger import save_context, Logger, CheckpointIO
from Tools import FLAGS, load_config, utils_torch

# from library import loss_gan
from library import inputs, data_iters

KEY_ARGUMENTS = load_config(FLAGS.config_file)
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

img_size = 32
torch.manual_seed(1234)
torch.cuda.manual_seed(1235)
np.random.seed(1236)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS.device = device

print_interv, eval_interv = 20, 400
iters = data_iters.get_dataloader(infinity=False)

label_ind = defaultdict(list)
labels = iters.dataset.get_label()
for ind, y in enumerate(labels):
    label_ind[y].append(ind)

label1 = 1
label2 = 5
num_aug_per_ins = 1
label_img = defaultdict(list)
for y in [label1, label2]:
    subset = torch.utils.data.Subset(iters.dataset, label_ind[y])
    loader = torch.utils.data.DataLoader(subset, 100, drop_last=False, shuffle=False, num_workers=256)
    for _ in range(num_aug_per_ins):
        for x, _, _ in loader:
            label_img[y].append(x)

x1 = torch.cat(label_img[label1], 0).reshape(-1, 3 * img_size * img_size)
x2 = torch.cat(label_img[label2], 0).reshape(-1, 3 * img_size * img_size)


def dxy(list1, list2, same=False):
    list1 = list1.unsqueeze(0)
    list2 = list2.unsqueeze(1)
    dist = torch.sum((list1 - list2) ** 2, -1)
    if same is True:
        dist = dist + torch.eye(dist.shape[0]) * 10000
    return torch.min(dist, 1)


print("d11", dxy(x1, x1).mean(0))
print("d12", dxy(x1, x2).mean(0))
print("d22", dxy(x2, x2).mean(0))

