from library.models.modules import mlpModule, ByolWrapper, EbmWrapper

# from library.models.resnet_cifar import model_dict as cifar_resnet_model_dict
from library.models.resnet_cifar import model_dict as cifar_resnet_model_dict

# from library.models.resnet_cifar_ebm import model_dict as cifar_resnet_model_ebm_dict

# from library.models.resnet_imagenet import model_dict as imagenet_resnet_model_dict
# from library.models.mlp_toy import model_dict as toy_mlp_model_dict

from torch import nn
from Tools.utils_torch import Identity

all_models = dict()
for k in cifar_resnet_model_dict:
    all_models["cifar_" + k] = cifar_resnet_model_dict[k]

# for k in cifar_resnet_model_ebm_dict:
#     all_models["cifar_" + k] = cifar_resnet_model_ebm_dict[k]

# for k in imagenet_resnet_model_dict:
#     all_models["imagenet_" + k] = imagenet_resnet_model_dict[k]

# for k in toy_mlp_model_dict:
#     all_models["toy_" + k] = toy_mlp_model_dict[k]


def get_models(name):
    return all_models[name]


def get_norm_layer(name):
    return eval(name)
