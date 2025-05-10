import torch
import torch.nn as nn
import torch.nn.functional as F
from . import module as md
from attention.module import Activation
from typing import Optional, Union
from attention.block import DANetHead

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        # 拼接两个输入，其中一个经过上采样
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        #输出特征图的大小不会改变，因为卷积核的尺寸和填充设置保持了输入和输出的空间尺寸一致。
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        #实例化Attention
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")    #双线性插值方法将特征图的尺寸放大两倍
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, patch_size=10, activation="logsoftmax",
                     upsampling=1, att_depth=3):
        super().__init__()
        self.patch_size = patch_size
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        self.out_channels = out_channels

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size),
                                    stride=(self.patch_size, self.patch_size))

        self.activation = Activation(activation)
        self.att_depth = att_depth

    def forward(self, x, attentions):

        #将特征图尺寸缩小为原先的1/2**self.att_depth
        conv_feamap_size = nn.Conv2d(self.out_channels,self.out_channels, kernel_size=(2**self.att_depth, 2**self.att_depth),stride=(2**self.att_depth, 2**self.att_depth),groups=self.out_channels,bias=False)
        #创建了一个权重全为 1 的张量作为卷积核的可学习参数，以便后续调整
        conv_feamap_size.weight=nn.Parameter(torch.ones((self.out_channels, 1, 2**self.att_depth, 2**self.att_depth)))
        conv_feamap_size.to(x.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        x = self.conv_x(x)
        #上采样，放大特征图
        x = self.upsampling(x)
        #将一系列的二维切片按照给定的参数进行重叠区域的累积，从而还原为原始的二维图像。
        fold_layer = torch.nn.Fold(output_size=(x.size()[-2], x.size()[-1]), kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        correction=[]
        #找到张量 x 在指定维度上的最大值的索引
        x_argmax=torch.argmax(x, dim=1)
        #创建了一个与输入张量 x 具有相同形状的全零张量
        pr_temp = torch.zeros(x.size()).to(x.device)
        #创建了一个与输入张量 x 具有相同形状的全一张量
        src = torch.ones(x.size()).to(x.device)

        #x_softmax 张量中的每个通道上只有一个元素的值为1，根据最大值x_argmax索引的位置，
        # 将src分配到pr_temp中的对应位置，其他位置仍为0，最终x_softmax生成独热编码
        x_softmax = pr_temp.scatter(dim=1, index=x_argmax.unsqueeze(1), src=src)
        #卷积操作对输入进行滤波，获得特征图的每个区域的重要性权重。
        #归一化，以确保权重的总和为 1。
        argx_feamap = conv_feamap_size(x_softmax) / (2 ** self.att_depth * 2 ** self.att_depth)

        for i in range(x.size()[1]):
            #计算了特定通道上注意力张量 attentions 中非零元素的数量，并且在结果上添加了一个微小的值（0.00001）。
            non_zeros = torch.unsqueeze(torch.count_nonzero(attentions[:, i:i + 1, :, :], dim=-1) + 0.00001, dim=-1)
            #进行了矩阵乘法操作，将注意力权重与局部特征进行相乘，得到的结果将被用于修正输入特征图。
            att = torch.matmul(attentions[:, i:i + 1, :, :]/non_zeros, torch.unsqueeze(self.unfold(argx_feamap[:, i:i + 1, :, :]), dim=1).transpose(-1, -2))
            #去除索引维度
            #print("att.shape:", att.shape)
            att=torch.squeeze(att, dim=1)
            #通过transpose(-1, -2)交换最后两个维度的顺序，然后通过fold_layer将张量展平为指定的形状
            att = fold_layer(att.transpose(-1, -2))
            #将处理后的注意力张量att添加到correction列表中
            correction.append(att)
        #将correction列表中的张量沿着指定的维度1进行拼接，形成一个新的张量。
        correction=torch.cat(correction, dim=1)
        #这行代码将特征图x与修正的特征图correction相乘，并将结果与原始特征图相加，以获得最终的输出特征图。
        x = correction * x + x
        #对特征图x应用激活函数，以产生最终的输出结果。
        x = self.activation(x)

        return x, attentions


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
            in_channels=777,
            out_classes=777,
            activation: Optional[Union[str, callable]] = None,
            kernel_size=3,
            att_depth=3,
    ):
        super().__init__()

        seg_input_channels=in_channels
        seg_output_channels=out_classes

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        self.segmentation = SegmentationHead(in_channels=seg_input_channels, out_channels=seg_output_channels,
                                             kernel_size=kernel_size, activation=activation, att_depth=att_depth)

        # upsampling

        # 1024,32,32
        self.up_concat4 = unetUp(3072, 1024)
        # 512,64,64
        self.up_concat3 = unetUp(1536, 512)
        # 256,128,128
        self.up_concat2 = unetUp(768, 256)
        # 64,256,256
        self.up_concat1 = unetUp(320, 64)

        self.DANet2 = DANetHead(256, 256)
        self.DANet3= DANetHead(512, 512)
        self.DANet4 = DANetHead(1024, 1024)
        self.DANet5 = DANetHead(2048, 2048)

    def forward(self, features, attentions):

        features[2] = self.DANet2(features[2])
        features[3] = self.DANet3(features[3])
        features[4] = self.DANet4(features[4])
        #features[5] = self.DANet5(features[5])


        features = features[1:]
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        x, attentions = self.segmentation(x, attentions)

        return x, attentions

