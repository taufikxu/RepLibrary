# from library.models.resnet_cifar import model_dict as cifar_resnet_model_dict
from library.models.resnet_cifar import model_dict as cifar_resnet_model_dict

all_models = dict()
for k in cifar_resnet_model_dict:
    all_models["cifar_" + k] = cifar_resnet_model_dict[k]


def get_models(name):
    return all_models[name]
