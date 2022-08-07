import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .layers.activations import act_layers
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
import os

model_urls = {
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",  # noqa: E501
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",  # noqa: E501
    "shufflenetv2_1.5x": None,
    "shufflenetv2_2.0x": None,
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, activation="ReLU"):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                act_layers(activation),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_size="1.5x",
        out_stages=(2, 3, 4),
        with_last_conv=False,
        kernal_size=3,
        activation="ReLU",
        pretrain=True,
    ):
        super(ShuffleNetV2, self).__init__()
        # out_stages can only be a subset of (2, 3, 4)
        assert set(out_stages).issubset((2, 3, 4))

        print("model size is ", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        self.activation = activation
        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            act_layers(activation),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]
        ):
            seq = [
                ShuffleV2Block(
                    input_channels, output_channels, 2, activation=activation
                )
            ]
            for i in range(repeats - 1):
                seq.append(
                    ShuffleV2Block(
                        output_channels, output_channels, 1, activation=activation
                    )
                )
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                act_layers(activation),
            )
            self.stage4.add_module("conv5", conv5)
        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _initialize_weights(self, pretrain=True):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
        if isinstance(pretrain, str) and os.path.exists(pretrain):
            pretrained_state_dict = torch.load(pretrain, map_location='cpu')
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            url = model_urls["shufflenetv2_{}".format(self.model_size)]
            if url is not None:
                pretrained_state_dict = model_zoo.load_url(url)
                print("=> loading pretrained model {}".format(url))
                self.load_state_dict(pretrained_state_dict, strict=False)


class ShufflenetV2Backbone(Backbone):
    def __init__(self, cfg=None, out_features=[2, 3, 4], pretrain=True):
        super(ShufflenetV2Backbone, self).__init__()
        self.model = ShuffleNetV2(
            model_size="1.0x", pretrain=pretrain, out_stages=out_features
        )

        self.return_features_num_channels = []
        self.out_features = out_features

    def forward(self, x):
        res = self.model(x)
        # we return list now, not dictionary
        return res


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_backbone(cfg, input_shape):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    weights_f = cfg.MODEL.BACKBONE.WEIGHTS
    assert set(out_features).issubset((2, 3, 4))

    out_feature_channels = {"stage2": 24, "stage3": 32, "stage4": 96, "stage5": 320}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    model = ShufflenetV2Backbone(cfg, out_features=out_features, pretrain=weights_f)
    model._out_features = out_features
    model._out_feature_channels = {2: 116, 3: 232, 4: 464, 5: 320}
    model._out_feature_strides = {2: 4, 3: 8, 4: 16, 5: 32}
    return model


if __name__ == "__main__":
    import sys

    backbone = ShufflenetV2Backbone()

    a = torch.randn([1, 3, 512, 512])
    o = backbone(a)

    # mobilenetv2 channels are weired.
    """
    res2 torch.Size([1, 24, 128, 128])
    res3 torch.Size([1, 32, 64, 64])
    res4 torch.Size([1, 96, 32, 32])
    res5 torch.Size([1, 320, 16, 16])
    """

    for k, v in o.items():
        print(k, v.shape)
