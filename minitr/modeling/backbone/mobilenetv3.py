from torch import nn
from torch.nn import BatchNorm2d
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from torchvision.models.mobilenetv3 import mobilenet_v3_small, mobilenet_v3_large
import torch


class MobileNetV3(Backbone):
    def __init__(self, type="s", out_features=["res3", "res4", "res5"]):
        super(MobileNetV3, self).__init__()
        if type == "l":
            self.model = mobilenet_v3_large(pretrained=True)
            self.return_features_indices = [3, 6, 10, 15]
        else:
            self.model = mobilenet_v3_small(pretrained=True)
            self.return_features_indices = [1, 3, 7, 10]

        self.out_features = out_features
        """
        torch.Size([1, 16, 256, 256])
        torch.Size([1, 16, 128, 128])
        torch.Size([1, 24, 64, 64])
        torch.Size([1, 24, 64, 64])
        torch.Size([1, 40, 32, 32])
        torch.Size([1, 40, 32, 32])
        torch.Size([1, 40, 32, 32])
        torch.Size([1, 48, 32, 32])
        torch.Size([1, 48, 32, 32])
        torch.Size([1, 96, 16, 16])
        torch.Size([1, 96, 16, 16])
        torch.Size([1, 96, 16, 16])
        torch.Size([1, 576, 16, 16])
        """

        self.return_features_num_channels = []
        self.features = self.model.features

    def forward(self, x):
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.return_features_indices:
                res.append(x)
        all_res = {"res{}".format(i + 2): r for i, r in enumerate(res)}
        return {k: v for k, v in all_res.items() if k in self.out_features}

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2.0 / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


@BACKBONE_REGISTRY.register()
def build_mobilenetv3_small_backbone(cfg, input_shape):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES

    out_feature_channels = {"res2": 16, "res3": 24, "res4": 48, "res5": 96}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    model = MobileNetV3(type="s", out_features=out_features)
    model._out_features = out_features
    model._out_feature_channels = {
        k: v for k, v in out_feature_channels.items() if k in out_features
    }
    model._out_feature_strides = {
        k: v for k, v in out_feature_strides.items() if k in out_features
    }
    return model


@BACKBONE_REGISTRY.register()
def build_mobilenetv3_large_backbone(cfg, input_shape):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES

    out_feature_channels = {"res2": 24, "res3": 40, "res4": 80, "res5": 160}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    model = MobileNetV3(type="l", out_features=out_features)
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

    backbone = MobileNetV3(type="s")

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
