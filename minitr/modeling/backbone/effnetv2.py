import copy
from dataclasses import dataclass
from functools import partial
import math
from typing import Callable, List, Optional, Any, Sequence, Tuple, Union
import torch
from torch import Tensor, nn
from timm.models.layers import DropPath
from torch.nn.functional import cross_entropy, dropout, one_hot, softmax

try:
    from torchvision.models.efficientnet import (
        _efficientnet,
        _MBConvConfig,
        MBConvConfig,
        MBConv,
    )
    from torchvision.ops.misc import (
        Conv2dNormActivation,
        SqueezeExcitation,
        ConvNormActivation,
    )
except ImportError:
    _MBConvConfig = object
    MBConvConfig = object
from torchvision.ops import StochasticDepth

from detectron2.layers import ShapeSpec
from detectron2.modeling import (
    BACKBONE_REGISTRY,
    RPN_HEAD_REGISTRY,
    Backbone,
    build_anchor_generator,
)
from yolov7.utils.checkpoint import load_checkpoint
from alfred import logger
import os


pretrained_weights_map = {
    "efficientnetv2_s": "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
    "efficientnetv2_m": "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
}


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class FusedMBConv(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


def _efficientnet_conf(
    arch: str,
    **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        bneck_conf = partial(
            MBConvConfig,
            width_mult=kwargs.pop("width_mult"),
            depth_mult=kwargs.pop("depth_mult"),
        )
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


class EfficientNetV2Backbone(Backbone):
    def __init__(
        self,
        weight_f="",
        type="s",
        out_features=None,
        stochastic_depth_prob: float = 0.2,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = partial(
            nn.BatchNorm2d, eps=1e-03
        ),
    ):
        super(EfficientNetV2Backbone, self).__init__()

        if type == "s":
            inverted_residual_setting, last_channel = _efficientnet_conf(
                "efficientnet_v2_s"
            )
        else:
            ValueError("only s supported now.")

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[_MBConvConfig]"
            )

        # if block is None:
        #     block = MBConv
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.stages: List[nn.Module] = []
        self.shape_spec_dict = {}

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.stages.append(
            ConvNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for i, cnf in enumerate(inverted_residual_setting):
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            # layers.append(nn.Sequential(*stage))
            self.stages.append(nn.Sequential(*stage))
            self.shape_spec_dict[f"stage{i+1}"] = ShapeSpec(
                channels=inverted_residual_setting[i].out_channels,
                stride=inverted_residual_setting[i].stride,
            )

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        self.stages.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.stages = nn.Sequential(*self.stages)
        self.out_features = out_features
        self.output_shape_dict = {}

        for i in range(len(self.stages)):
            name = "stage{}".format(i)
            if name in out_features:
                self.output_shape_dict[name] = self.shape_spec_dict[name]

        self._load_pretrained(weight_f)

    def _load_pretrained(self, pretrained):
        logger.info("start init pretrained weights.")

        def _init_weights(m):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        if isinstance(pretrained, str):
            logger.info("getting checkpoints from file or url..")
            state_dict = load_checkpoint(
                self,
                pretrained,
                strict=False,
                revise_keys=[("features", "stages")],
                force_return=False,
            )

        elif pretrained is None:
            logger.info("init default weights.")
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    # return features for each stage
    def forward(self, x):
        features = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if f"stage{i}" in self.out_features:
                features[f"stage{i}"] = x
        return features

    def output_shape(self):
        return self.output_shape_dict


@BACKBONE_REGISTRY.register()
def build_efficientnetv2_s_backbone(cfg, input_shape=None):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES

    weights_f = cfg.MODEL.BACKBONE.WEIGHTS
    if not os.path.exists(weights_f):
        weights_f = pretrained_weights_map["efficientnetv2_s"]

    backbone = EfficientNetV2Backbone(
        weight_f=weights_f, type="s", out_features=out_features
    )
    return backbone


if __name__ == "__main__":
    import sys

    a = sys.argv[1]

    backbone = EfficientNetV2Backbone(
        weight_f=a, type="s", out_features=["stage3", "stage4", "stage6"]
    )

    a = torch.randn([1, 3, 512, 512])
    o = backbone(a)

    for k, v in o.items():
        print(k, v.shape)
    # print(backbone)
    # for k, v in backbone.state_dict().items():
    #     print(k, v.shape)
