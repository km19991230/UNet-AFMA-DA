import functools
import torch.utils.model_zoo as model_zoo
from .encoder_channelatt_img import encoders_channelatt_img


from .model import SegmentationModel
from .head import (SegmentationHead, ClassificationHead)

from .module import (
    Conv2dReLU,
    Attention
)

encoders = {}
encoders.update(encoders_channelatt_img)

def get_encoder(name, in_channels=3, classes_num=4, depth=5, att_depth=3, weights=None, replace_stride_with_dilation=None):

    try:
        Encoder = encoders[name]["encoder"]

    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported utils: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    params.update(att_depth=att_depth)
    #params.update(replace_stride_with_dilation=replace_stride_with_dilation)
    params.update(classes_num=classes_num)
    encoder = Encoder(**params)



    encoder.set_in_channels(in_channels)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    settings = encoders[encoder_name]["pretrained_settings"]

    if pretrained not in settings.keys():
        raise ValueError("Available pretrained options {}".format(settings.keys()))

    formatted_settings = {}
    formatted_settings["input_space"] = settings[pretrained].get("input_space")
    formatted_settings["input_range"] = settings[pretrained].get("input_range")
    formatted_settings["mean"] = settings[pretrained].get("mean")
    formatted_settings["std"] = settings[pretrained].get("std")
    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    # print(params)
    # {'input_space': 'RGB', 'input_range': [0, 1], 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    return functools.partial(preprocess_input, **params)

def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

if __name__ == '__main__':
    for name in encoders:
        print(name)
