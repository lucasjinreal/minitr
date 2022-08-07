import torch
import torch.nn as nn
from detectron2.layers import ShapeSpec
import torch.nn.functional as F
from ..backbone.layers.utils import get_norm
from ..backbone.layers.wrappers import get_activation
from yolov7.utils.initializer import normal_init
from yolov7.utils.misc import autopad

class ConvBNLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        filter_size=3,
        stride=1,
        groups=1,
        padding=0,
        norm_type="BN",
        norm_decay=0.0,
        act="silu",
        freeze_norm=False,
        data_format="NCHW",
        name="",
    ):
        """
        conv + bn + activation layer

        Args:
            ch_in (int): input channel
            ch_out (int): output channel
            filter_size (int): filter size, default 3
            stride (int): stride, default 1
            groups (int): number of groups of conv layer, default 1
            padding (int): padding size, default 0
            norm_type (str): batch norm type, default bn
            norm_decay (str): decay for weight and bias of batch norm layer, default 0.
            act (str): activation function type, default 'leaky', which means leaky_relu
            freeze_norm (bool): whether to freeze norm, default False
            data_format (str): data format, NCHW or NHWC
        """
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            # data_format=data_format,
            bias=False,
        )
        self.batch_norm = get_norm(norm_type, out_channels=ch_out)
        self.act = get_activation(name=act)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        out = self.act(out)
        return out


class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        norm_type,
        freeze_norm=False,
        act="leaky",
        data_format="NCHW",
    ):
        """
        SPP layer, which consist of four pooling layer follwed by conv layer

        Args:
            ch_in (int): input channel of conv layer
            ch_out (int): output channel of conv layer
            k (int): kernel size of conv layer
            norm_type (str): batch norm type
            freeze_norm (bool): whether to freeze norm, default False
            name (str): layer name
            act (str): activation function
            data_format (str): data format, NCHW or NHWC
        """
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format

        self.pool = nn.Sequential(
            *[nn.MaxPool2d(
                kernel_size=size,
                stride=1,
                padding=size // 2,
                ceil_mode=False,
            ) for size in pool_size]
        )
        self.conv = ConvBNLayer(
            ch_in,
            ch_out,
            k,
            padding=k // 2,
            norm_type=norm_type,
            freeze_norm=freeze_norm,
            act=act,
            data_format=data_format,
        )

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == "NCHW":
            y = torch.concat(outs, axis=1)
        else:
            y = torch.concat(outs, axis=-1)

        y = self.conv(y)
        return y



class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, pool_size=(5, 9, 13)):
        super(SPPCSPC, self).__init__()

        print(c1, c2)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in pool_size]
        )
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class GhostSPPCSPC(SPPCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, pool_size=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, pool_size)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob, data_format="NCHW"):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1.0 - self.keep_prob) / (self.block_size**2)
            if self.data_format == "NCHW":
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = torch.rand(x.shape, device=x.device)
            matrix = (matrix < gamma).float()
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
            )
            mask = 1.0 - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class PPYOLODetBlockCSP(nn.Module):
    def __init__(self, cfg, ch_in, ch_out, act, norm_type, name, data_format="NCHW"):
        """
        PPYOLODetBlockCSP layer

        Args:
            cfg (list): layer configs for this block
            ch_in (int): input channel
            ch_out (int): output channel
            act (str): default mish
            name (str): block name
            data_format (str): data format, NCHW or NHWC
        """
        super(PPYOLODetBlockCSP, self).__init__()
        self.data_format = data_format
        self.conv1 = ConvBNLayer(
            ch_in,
            ch_out,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name + "left",
            data_format=data_format,
        )
        self.conv2 = ConvBNLayer(
            ch_in,
            ch_out,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name + "right",
            data_format=data_format,
        )
        self.conv3 = ConvBNLayer(
            ch_out * 2,
            ch_out * 2,
            1,
            padding=0,
            act=act,
            norm_type=norm_type,
            name=name,
            data_format=data_format,
        )
        self.conv_module = nn.Sequential(
            *[layer(*args, **kwargs) for layer_name, layer, args, kwargs in cfg]
        )
        # for idx, (layer_name, layer, args, kwargs) in enumerate(cfg):
        #     # kwargs.update(name=name + layer_name, data_format=data_format)
        #     self.conv_module.add_module(layer_name, layer(*args, **kwargs))

    def forward(self, inputs):
        conv_left = self.conv1(inputs)
        conv_right = self.conv2(inputs)
        conv_left = self.conv_module(conv_left)
        if self.data_format == "NCHW":
            conv = torch.concat([conv_left, conv_right], axis=1)
        else:
            conv = torch.concat([conv_left, conv_right], axis=-1)

        conv = self.conv3(conv)
        return conv, conv


class DSPPAN(nn.Module):
    __shared__ = ["norm_type", "data_format"]

    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        norm_type="BN",
        data_format="NCHW",
        act="silu",
        conv_block_num=3,
        drop_block=False,
        block_size=3,
        keep_prob=0.9,
        spp=False,
        base_channels=512,
    ):
        """

        DSP PAN

        A neck with dropblock, spp, and pan connection.
        Same idea as PPYOLO but with some modifications.


        Args:
            in_channels (list): input channels for fpn
            norm_type (str): batch norm type, default bn
            data_format (str): data format, NCHW or NHWC
            act (str): activation function, default mish
            conv_block_num (int): conv block num of each pan block
            drop_block (bool): whether use DropBlock or not
            block_size (int): block size of DropBlock
            keep_prob (float): keep probability of DropBlock
            spp (bool): whether use spp or not

        """
        super(DSPPAN, self).__init__()
        assert len(in_channels) > 0, "in_channels length should > 0"
        self.in_channels = in_channels
        self.num_blocks = len(in_channels)
        # parse kwargs
        self.drop_block = drop_block
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.spp = spp
        self.conv_block_num = conv_block_num
        self.data_format = data_format
        self.base_channels = base_channels
        self.out_channels = []

        if self.drop_block:
            dropblock_cfg = [
                ["dropblock", DropBlock, [self.block_size, self.keep_prob], dict()]
            ]
        else:
            dropblock_cfg = []

        # fpn
        self.fpn_blocks = []
        self.fpn_routes = []
        fpn_channels = []
        for i, ch_in in enumerate(self.in_channels[::-1]):
            if i > 0:
                ch_in += base_channels // (2 ** (i - 1))
            channel = base_channels // (2**i)
            base_cfg = []
            for j in range(self.conv_block_num):
                base_cfg += [
                    # name, layer, args
                    [
                        "{}0".format(j),
                        ConvBNLayer,
                        [channel, channel, 1],
                        dict(padding=0, act=act, norm_type=norm_type),
                    ],
                    [
                        "{}1".format(j),
                        ConvBNLayer,
                        [channel, channel, 3],
                        dict(padding=1, act=act, norm_type=norm_type),
                    ],
                ]

            if i == 0 and self.spp:
                base_cfg[3] = [
                    "spp",
                    # SPP,
                    # SPPCSPC,
                    GhostSPPCSPC,
                    # [channel * 4, channel, 1],
                    [channel, channel, 1],
                    # dict(pool_size=[5, 9, 13], act=act, norm_type=norm_type),
                    dict(pool_size=[5, 9, 13]),
                ]

            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = "fpn{}".format(i)
            fpn_block = PPYOLODetBlockCSP(
                cfg, ch_in, channel, act, norm_type, name, data_format
            )
            self.add_module(name, fpn_block)
            self.fpn_blocks.append(fpn_block)
            fpn_channels.append(channel * 2)
            if i < self.num_blocks - 1:
                name = "fpn_transition_{}".format(i)
                route = ConvBNLayer(
                    ch_in=channel * 2,
                    ch_out=channel,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    act=act,
                    norm_type=norm_type,
                    data_format=data_format,
                    name=name,
                )
                self.add_module(name, route)
                self.fpn_routes.append(route)
        # pan
        self.pan_blocks = []
        self.pan_routes = []
        self._out_channels = [
            base_channels // (2 ** (self.num_blocks - 2)),
        ]
        for i in reversed(range(self.num_blocks - 1)):
            name = "pan_transition{}".format(i)
            route = ConvBNLayer(
                ch_in=fpn_channels[i + 1],
                ch_out=fpn_channels[i + 1],
                filter_size=3,
                stride=2,
                padding=1,
                act=act,
                norm_type=norm_type,
                data_format=data_format,
                name=name,
            )
            self.add_module(name, route)
            self.pan_routes = [
                route,
            ] + self.pan_routes
            base_cfg = []
            ch_in = fpn_channels[i] + fpn_channels[i + 1]
            channel = base_channels // (2**i)
            for j in range(self.conv_block_num):
                base_cfg += [
                    # name, layer, args
                    [
                        "{}0".format(j),
                        ConvBNLayer,
                        [channel, channel, 1],
                        dict(padding=0, act=act, norm_type=norm_type),
                    ],
                    [
                        "{}1".format(j),
                        ConvBNLayer,
                        [channel, channel, 3],
                        dict(padding=1, act=act, norm_type=norm_type),
                    ],
                ]

            cfg = base_cfg[:4] + dropblock_cfg + base_cfg[4:]
            name = "pan{}".format(i)
            pan_block = PPYOLODetBlockCSP(
                cfg, ch_in, channel, act, norm_type, name, data_format
            )
            self.add_module(name, pan_block)

            self.pan_blocks = [
                pan_block,
            ] + self.pan_blocks
            self._out_channels.append(channel * 2)

        self._out_channels = self._out_channels[::-1]
        self.out_channels = self._out_channels

    def forward(self, blocks, for_mot=False):
        assert len(blocks) == self.num_blocks
        blocks = blocks[::-1]
        fpn_feats = []

        # add embedding features output for multi-object tracking model
        if for_mot:
            emb_feats = []

        # print(self.fpn_blocks)
        for i, block in enumerate(blocks):
            if i > 0:
                if self.data_format == "NCHW":
                    block = torch.concat([route, block], axis=1)
                else:
                    block = torch.concat([route, block], axis=-1)
            from alfred import print_shape

            # print_shape(block)
            route, tip = self.fpn_blocks[i](block)
            # print_shape(tip)
            fpn_feats.append(tip)

            if for_mot:
                # add embedding features output
                emb_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.upsample(route, scale_factor=2.0, mode="nearest")

        pan_feats = [
            fpn_feats[-1],
        ]
        route = fpn_feats[self.num_blocks - 1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            if self.data_format == "NCHW":
                block = torch.concat([route, block], axis=1)
            else:
                block = torch.concat([route, block], axis=-1)

            route, tip = self.pan_blocks[i](block)
            pan_feats.append(tip)

        if for_mot:
            return {"yolo_feats": pan_feats[::-1], "emb_feats": emb_feats}
        else:
            return pan_feats[::-1]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_channels": [i.channels for i in input_shape],
        }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
