""" 
An implementation of TinyNet
Requirements: timm==0.1.20
"""
from timm.models.efficientnet_builder import *
from timm.models.efficientnet import EfficientNet, EfficientNetFeatures, _cfg
from timm.models.registry import register_model
from timm.models.layers.activations import Swish
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone


def _gen_tinynet(
    variant_cfg,
    channel_multiplier=1.0,
    depth_multiplier=1.0,
    depth_trunc="round",
    **kwargs,
):
    """Creates a TinyNet model."""
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"],
        ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"],
        ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc=depth_trunc),
        num_features=max(1280, round_channels(1280, channel_multiplier, 8, None)),
        stem_size=32,
        fix_stem=True,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = EfficientNet(**model_kwargs)
    model.default_cfg = variant_cfg
    return model


def tinynet(r=1.0, w=1.0, d=1.0, **kwargs):
    """TinyNet"""
    hw = int(224 * r)
    model = _gen_tinynet(
        _cfg(input_size=(3, hw, hw)), channel_multiplier=w, depth_multiplier=d, **kwargs
    )
    return model


class TinyNet(Backbone):
    def __init__(self, cfg=None, out_features=["res3", "res4", "res5"]):
        super(TinyNet, self).__init__()
        self.model = tinynet(pretrained=True)

        self.return_features_indices = [3, 6, 13, 17]
        self.return_features_num_channels = []
        self.features = self.model.features
        self.out_features = out_features

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.return_features_indices:
                res.append(x)
        all_res = {"res{}".format(i + 2): r for i, r in enumerate(res)}
        return {k: v for k, v in all_res.items() if k in self.out_features}


@BACKBONE_REGISTRY.register()
def build_tinynet_backbone(cfg, input_shape):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES

    out_feature_channels = {"res2": 24, "res3": 32, "res4": 96, "res5": 320}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    model = TinyNet(cfg, out_features=out_features)
    model._out_features = out_features
    model._out_feature_channels = {
        k: v for k, v in out_feature_channels.items() if k in out_features
    }
    model._out_feature_strides = {
        k: v for k, v in out_feature_strides.items() if k in out_features
    }
    return model


if __name__ == "__main__":
    tn = tinynet()
    print(tn)
