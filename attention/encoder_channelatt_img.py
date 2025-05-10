from copy import deepcopy
import torch.nn as nn
import torch

from nets.resnet import ResNet, Bottleneck
from attention.encoderMixin import EncoderMixin

import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class Encoder_channelatt_img(ResNet, EncoderMixin):
    def __init__(self, out_channels, classes_num=4, patch_size=10, depth=5, att_depth=3, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._attention_on_depth = att_depth

        self._out_channels = out_channels

        self._in_channels = 3

        self.patch_size = patch_size

        self.conv_img = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), padding=3),
            nn.Conv2d(64, classes_num, kernel_size=(3, 3), padding=1)
        )

        self.conv_feamap = nn.Sequential(
            nn.Conv2d(self._out_channels[self._attention_on_depth], classes_num, kernel_size=(1, 1), stride=1)
        )

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                stride=(self.patch_size, self.patch_size))

        self.resolution_trans = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        attentions = []

        x = stages[0](x)

        features.append(x)

        ini_img = self.conv_img(x)

        x = stages[1](x)

        features.append(x)

        if self._attention_on_depth == 1:

            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)

        x = stages[2](x)

        features.append(x)

        if self._attention_on_depth == 2:

            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)

        x = stages[3](x)

        features.append(x)

        if self._attention_on_depth == 3:

            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat(attentions, dim=1)

        x = stages[4](x)

        features.append(x)

        if self._attention_on_depth == 4:

            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)

                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat(attentions, dim=1)

        x = stages[5](x)

        features.append(x)

        if self._attention_on_depth == 5:


            feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)

            for i in range(feamap.size()[1]):
                unfold_img = self.unfold(ini_img[:, i:i + 1, :, :]).transpose(-1, -2)
                unfold_img = self.resolution_trans(unfold_img)

                unfold_feamap = self.unfold(feamap[:, i:i + 1, :, :])
                unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)

                att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)

                att = torch.unsqueeze(att, 1)

                attentions.append(att)

            attentions = torch.cat((attentions), dim=1)

        return features, attentions

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, strict=False, **kwargs)


def resnet50(pretrained=True, **kwargs):
    encoder = encoders_channelatt_img["resnet50"]["encoder"]
    model = encoder(out_channels=(3, 64, 256, 512, 1024, 2048), classes_num=2, patch_size=5, depth=5, att_depth=3).train()

    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'),
            strict=False)

    return model

encoders_channelatt_img = {
    "resnet50": {
        "encoder": Encoder_channelatt_img,

        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
}
