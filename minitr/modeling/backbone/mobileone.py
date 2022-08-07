"""
Implementation backbone of MobileOne

An Improved One millisecond Mobile Backbone
"""
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
import torch
from alfred import logger
from yolov7.utils.checkpoint import load_checkpoint

try:
    # from nb.torch.backbones.mobileone import MobileOneNet
    from nb.torch.backbones.mobileone_apple import (
        MobileOne as MobileOneNet,
        MOBILEONE_PARAMS,
    )
except ImportError:
    logger.info("pip install nbnb to enable NeuralBlocksBuilder feature.")


def make_mobileone_s0(deploy=False):
    blocks = [1, 2, 8, 5, 5, 1]
    strides = [2, 2, 2, 2, 1, 2]
    ks = [4, 4, 4, 4, 4, 4] if deploy is False else [1, 1, 1, 1, 1, 1]
    width_muls = [0.75, 0.75, 1, 1, 1, 2]  # 261 M flops
    channels = [64, 64, 128, 256, 256, 512]
    # channels = [48, 48, 128, 256, 256, 512]
    # model = MobileOneNet(blocks, ks, channels, strides, width_muls, deploy=deploy)
    variant_params = MOBILEONE_PARAMS["s0"]
    model = MobileOneNet(inference_mode=deploy, **variant_params)
    return model


class MobileOne(Backbone):
    def __init__(
        self,
        type="s0",
        deploy=False,
        out_features=["stage2", "stage3", "stage4"],
        pretrained=None,
    ):
        super(MobileOne, self).__init__()
        self.deploy = deploy
        if type == "s0":
            self.model = make_mobileone_s0(deploy)
        else:
            logger.error("only s0 supported now.")

        if pretrained:
            logger.info(f"loading pretrained weights from: {pretrained}")
            load_checkpoint(self.model, pretrained, strict=False)

        self.return_features_num_channels = []
        self.out_features = out_features

    def forward(self, x):
        out = {}

        x0 = self.model.stage0(x)
        x1 = self.model.stage1(x0)
        out["stage1"] = x1
        x2 = self.model.stage2(x1)
        out["stage2"] = x2
        x3 = self.model.stage3(x2)
        out["stage3"] = x3
        x4 = self.model.stage4(x3)
        out["stage4"] = x4
        # for k, v in out.items():
        #     print(k, v.shape)
        return {k: v for k, v in out.items() if k in self.out_features}


"""
[oi]:  torch.Size([1, 128, 52, 80]) cpu torch.float32
[oi]:  torch.Size([1, 256, 26, 40]) cpu torch.float32
[oi]:  torch.Size([1, 256, 26, 40]) cpu torch.float32
[oi]:  torch.Size([1, 1024, 13, 20]) cpu torch.float32
"""


@BACKBONE_REGISTRY.register()
def build_mobileone_backbone(cfg, input_shape):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
    weights_f = cfg.MODEL.BACKBONE.WEIGHTS

    out_feature_channels = {"stage1": 48, "stage2": 128, "stage3": 256, "stage4": 1024}
    out_feature_strides = {"stage1": 4, "stage2": 8, "stage3": 16, "stage4": 32}

    if torch.onnx.is_in_onnx_export():
        logger.info("[onnx] in onnx export mode, mobileone will using deploy mode.")
        model = MobileOne(type="s0", deploy=True, out_features=out_features)
    else:
        model = MobileOne(type="s0", out_features=out_features, pretrained=weights_f)
    model._out_features = out_features
    model._out_feature_channels = {
        k: v for k, v in out_feature_channels.items() if k in out_features
    }
    model._out_feature_strides = {
        k: v for k, v in out_feature_strides.items() if k in out_features
    }
    return model


if __name__ == "__main__":
    import sys

    backbone = MobileOne()

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
