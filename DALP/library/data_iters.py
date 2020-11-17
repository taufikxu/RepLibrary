import random

import torch
from Tools import FLAGS
from Tools.utils_torch import infinity_loader
from torch import Tensor
from torchvision import datasets, transforms

dataset_info = {
    "cifar10": (32, 10, 50000),
    "imagenet": (224, 1000, 1281167),
}


class indexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index

    def get_label(self):
        return self.dataset.targets


def get_data_augmentation(aug=True):
    if FLAGS.dataset.lower() == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        normalize = transforms.Normalize(mean=mean, std=std)
        normalize = transforms.Compose([transforms.ToTensor(), normalize])

        if aug is False:
            aug = normalize
        else:
            aug = transforms.Compose([transforms.RandomCrop(32, padding=2), transforms.RandomHorizontalFlip(), normalize,])
    elif FLAGS.dataset.lower() == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        normalize = transforms.Compose([transforms.ToTensor(), normalize])

        if aug is False:
            aug = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), normalize])
        else:
            aug = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    transforms.RandomApply([transforms.GaussianBlur(23, [0.1, 2.0])], p=1.0),
                    normalize,
                ]
            )
    else:
        raise ValueError("Unknown data augmentation")
    return aug


def get_dataset(train):
    if FLAGS.dataset.lower() == "cifar10":
        transf = get_data_augmentation(train)
        sets = datasets.CIFAR10("/home/LargeData/Regular/cifar", train=train, download=True, transform=transf,)
    elif FLAGS.dataset.lower() == "imagenet":
        transf = get_data_augmentation(train)
        split = "train" if train is True else "val"
        sets = datasets.ImageNet("/home/LargeData/Large/ImageNet", split=split, transform=transf,)
    else:
        raise ValueError("Unknown dataset")
    return indexedDataset(sets)


def get_dataloader(
    batch_size=None, dataset=None, train=True, train_aug=True, infinity=True,
):
    if batch_size is None:
        batch_size = FLAGS.batch_size
    if dataset is None:
        dataset = get_dataset(train)

    loader = torch.utils.data.DataLoader(dataset, batch_size, drop_last=True, shuffle=True, num_workers=16,)
    nimages = len(dataset)
    if train is True:
        assert nimages == dataset_info[FLAGS.dataset][2]
    if infinity is True:
        return infinity_loader(loader), nimages // batch_size
    else:
        return loader
