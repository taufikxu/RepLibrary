import torch
from torch import Tensor
from torchvision import datasets
from torchvision import transforms

from Tools import FLAGS
from Tools.utils_torch import infinity_loader
import numpy as np
import PIL.ImageOps, PIL.Image

dataset_info = {
    "cifar10": (32, 10, 50000),
    "imagenet": (224, 1000, 1281167),
}
dataset_shape = {
    "cifar10": (3, 32, 32),
    "imagenet": (3, 224, 224),
}


class SolarizeImage(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if isinstance(img, Tensor):
            mask = (img < 0.5).float()
            result = mask * img + (1.0 - mask) * (1.0 - img)
            return result
        else:
            return PIL.ImageOps.solarize(img)

    def __repr__(self):
        return self.__class__.__name__


class TwoCropsTransform(torch.nn.Module):
    def __init__(self, transf_view1, transf_view2):
        super().__init__()
        self.view1 = transf_view1
        self.view2 = transf_view2

    def forward(self, x):
        x1 = self.view1(x)
        x2 = self.view2(x)
        if len(x1.shape) > 2:
            return torch.cat([x1, x2], -3)
        else:
            return torch.cat([x1, x2], -1)


def get_data_augmentation(aug=True, toTensor=False):
    if FLAGS.dataset.lower() == "cifar10":

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]

        normalize = transforms.Normalize(mean=mean, std=std)
        if toTensor is True:
            normalize = transforms.Compose([transforms.ToTensor(), normalize])

        if aug is False:
            return normalize

        view1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(3, [0.1, 2.0])], p=1.0),
                transforms.RandomApply([SolarizeImage()], p=0.0),
                normalize,
            ]
        )
        view2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(3, [0.1, 2.0])], p=0.1),
                transforms.RandomApply([SolarizeImage()], p=0.2),
                normalize,
            ]
        )
        augment = TwoCropsTransform(view1, view2)
        return augment
    elif FLAGS.dataset.lower() == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        if toTensor is True:
            normalize = transforms.Compose([transforms.ToTensor(), normalize])

        if aug is False:
            return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), normalize])

        view1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(23, [0.1, 2.0])], p=1.0),
                transforms.RandomApply([SolarizeImage()], p=0.0),
                normalize,
            ]
        )
        view2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(23, [0.1, 2.0])], p=0.1),
                transforms.RandomApply([SolarizeImage()], p=0.2),
                normalize,
            ]
        )
        augment = TwoCropsTransform(view1, view2)
        return augment
    else:
        raise ValueError("Unknown Dataset Name")


def get_dataset(train, train_aug=True, additional_data=False):
    if FLAGS.dataset.lower() == "cifar10":
        transf = get_data_augmentation(train_aug, toTensor=True)
        sets = datasets.CIFAR10("/home/LargeData/Regular/cifar", train=train, download=True, transform=transf,)
    elif FLAGS.dataset.lower() == "imagenet":
        transf = get_data_augmentation(train_aug, toTensor=True)
        split = "train" if train is True else "val"
        sets = datasets.ImageNet("/home/LargeData/Large/ImageNet", split=split, transform=transf,)
    else:
        transf = get_data_augmentation(train_aug)
        n_samples = 50000 if train is True else 1000
        sets = dataset_toy.ToyData(FLAGS.dataset, n_samples=n_samples, transform=transf, additional_data=additional_data,)
    return sets


def get_dataloader(
    batch_size=None, dataset=None, train=True, train_aug=True, infinity=True, additional_data=False,
):
    if batch_size is None:
        batch_size = FLAGS.batch_size
    if dataset is None:
        dataset = get_dataset(train, train_aug, additional_data)

    loader = torch.utils.data.DataLoader(dataset, batch_size, drop_last=True, shuffle=True, num_workers=16,)
    nimages = len(dataset)
    if train is True:
        assert nimages == dataset_info[FLAGS.dataset][2]
    if infinity is True:
        return infinity_loader(loader), nimages // batch_size
    else:
        return loader


def test_func():
    img = PIL.Image.open("./ADebug/tmp.png")
    imgt = transforms.ToTensor()(img)
    # print(img, imgt)

    seed1, seed2, seed3 = 1, 2, 3

    for seed1 in range(1, 200):
        torch.manual_seed(seed1)
        torch.cuda.manual_seed(seed2)
        np.random.seed(seed3)
        imgt.requires_grad_(True)
        transf_tensor = get_data_augmentation(toTensor=False)
        tensor_aug = transf_tensor(imgt)[:3, :, :]
        tensor_grad = torch.autograd.grad(torch.sum(tensor_aug), imgt)[0]
        print(torch.sum(tensor_aug), torch.sum(tensor_grad))

        # transforms.ToPILImage()(tensor_aug).save("tensor.png")

        # torch.manual_seed(seed1)
        # torch.cuda.manual_seed(seed2)
        # np.random.seed(seed3)
        # transf_pil = get_data_augmentation(toTensor=True)
        # img_aug = transf_pil(img)[:3, :, :]
        # transforms.ToPILImage()(img_aug).save("tensor_pil.png")
        # print(torch.sum(torch.abs(img_aug - tensor_aug)))

