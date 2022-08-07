"""
Implementation backbone of MobileOne

An Improved One millisecond Mobile Backbone
"""
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
import torch
from alfred import logger

try:
    from nb.torch.backbones.efficientformer import (
        EfficientFormer,
        EfficientFormer_depth,
        EfficientFormer_width,
    )
except ImportError:
    logger.info("pip install nbnb to enable NeurualBlocksBuilder feature.")


class efficientformer_l1_feat(EfficientFormer):
    def __init__(self, **kwargs):
        super().__init__(
            layers=EfficientFormer_depth["l1"],
            embed_dims=EfficientFormer_width["l1"],
            downsamples=[True, True, True, True],
            fork_feat=True,
            vit_num=1,
            **kwargs,
        )


class EfficientFormerBackbone(Backbone):
    def __init__(
        self,
        type="l1",
        deploy=False,
        out_features=["norm0", "norm2", "norm4", "norm6"],
        pretrained_weights=None,
    ):
        super(EfficientFormerBackbone, self).__init__()
        self.deploy = deploy
        if type == "l1":
            # self.model = efficientformer_l1(pretrained=False)
            self.model = efficientformer_l1_feat(
                init_cfg=dict(checkpoint=pretrained_weights)
            )
        else:
            logger.error("only l1 supported now.")

        self.return_features_indices = [3, 6, 13, 17]
        self.return_features_num_channels = []
        self.out_features = out_features

    def forward(self, x):
        o = self.model(x)
        # for k, v in out.items():
        #     print(k, v.shape)
        return {k: v for k, v in o.items() if k in self.out_features}


"""
[oi]:  torch.Size([1, 128, 52, 80]) cpu torch.float32
[oi]:  torch.Size([1, 256, 26, 40]) cpu torch.float32
[oi]:  torch.Size([1, 256, 26, 40]) cpu torch.float32
[oi]:  torch.Size([1, 1024, 13, 20]) cpu torch.float32
"""


@BACKBONE_REGISTRY.register()
def build_efficientformer_backbone(cfg, input_shape):
    pretrain_weights = cfg.MODEL.BACKBONE.WEIGHTS

    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES

    out_feature_channels = {"norm0": 48, "norm2": 96, "norm4": 224, "norm6": 448}
    out_feature_strides = {"norm0": 4, "norm2": 8, "norm4": 16, "norm6": 32}

    model = EfficientFormerBackbone(
        type="l1", out_features=out_features, pretrained_weights=pretrain_weights
    )
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

    backbone = EfficientFormerBackbone()

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
