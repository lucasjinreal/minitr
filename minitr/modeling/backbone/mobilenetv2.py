from torch import nn
from torch.nn import BatchNorm2d
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from torchvision.models.mobilenetv2 import mobilenet_v2
import torch


class MobileNetV2(Backbone):
    def __init__(self, cfg=None, out_features=["res3", "res4", "res5"]):
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=True)

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
def build_mobilenetv2_backbone(cfg, input_shape):
    out_features = cfg.MODEL.BACKBONE.OUT_FEATURES

    out_feature_channels = {"res2": 24, "res3": 32, "res4": 96, "res5": 320}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    model = MobileNetV2(cfg, out_features=out_features)
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

    backbone = MobileNetV2()

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
