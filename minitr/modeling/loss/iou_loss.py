import functools
import math
import torch
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import numpy as np

from yolov7.utils.boxes import bbox_iou
from yolov7.utils.boxes import bbox_overlaps
from yolov7.utils.misc import weighted_loss


class IouLoss(nn.Module):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(
        self, loss_weight=2.5, giou=False, diou=False, ciou=False, loss_square=True
    ):
        super(IouLoss, self).__init__()
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def forward(self, pbox, gbox):
        iou = bbox_iou(pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou


class IouAwareLoss(IouLoss):
    """
    iou aware loss, see https://arxiv.org/abs/1912.05992
    Args:
        loss_weight (float): iou aware loss weight, default is 1.0
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
    """

    def __init__(self, loss_weight=1.0, giou=False, diou=False, ciou=False):
        super(IouAwareLoss, self).__init__(
            loss_weight=loss_weight, giou=giou, diou=diou, ciou=ciou
        )

    def forward(self, ioup, pbox, gbox):
        iou = bbox_iou(pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        # iou.requires_grad = False
        iou = iou.detach()
        loss_iou_aware = F.binary_cross_entropy_with_logits(ioup, iou, reduction="none")
        loss_iou_aware = loss_iou_aware * self.loss_weight
        return loss_iou_aware


class MyIOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(MyIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        """
        输入矩形的格式是cx cy w h
        """
        assert pred.shape[0] == target.shape[0]

        boxes1 = pred
        boxes2 = target

        # 变成左上角坐标、右下角坐标
        boxes1_x0y0x1y1 = torch.cat(
            [boxes1[:, :2] - boxes1[:, 2:] * 0.5, boxes1[:, :2] + boxes1[:, 2:] * 0.5],
            dim=-1,
        )
        boxes2_x0y0x1y1 = torch.cat(
            [boxes2[:, :2] - boxes2[:, 2:] * 0.5, boxes2[:, :2] + boxes2[:, 2:] * 0.5],
            dim=-1,
        )

        # 两个矩形的面积
        boxes1_area = (boxes1_x0y0x1y1[:, 2] - boxes1_x0y0x1y1[:, 0]) * (
            boxes1_x0y0x1y1[:, 3] - boxes1_x0y0x1y1[:, 1]
        )
        boxes2_area = (boxes2_x0y0x1y1[:, 2] - boxes2_x0y0x1y1[:, 0]) * (
            boxes2_x0y0x1y1[:, 3] - boxes2_x0y0x1y1[:, 1]
        )

        # 相交矩形的左上角坐标、右下角坐标
        left_up = torch.maximum(boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2])
        right_down = torch.minimum(boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:])

        # 相交矩形的面积inter_area。iou
        inter_section = F.relu(right_down - left_up)
        inter_area = inter_section[:, 0] * inter_section[:, 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / (union_area + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            # 包围矩形的左上角坐标、右下角坐标
            enclose_left_up = torch.minimum(
                boxes1_x0y0x1y1[:, :2], boxes2_x0y0x1y1[:, :2]
            )
            enclose_right_down = torch.maximum(
                boxes1_x0y0x1y1[:, 2:], boxes2_x0y0x1y1[:, 2:]
            )

            # 包围矩形的面积
            enclose_wh = enclose_right_down - enclose_left_up
            enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

            giou = iou - (enclose_area - union_area) / enclose_area
            # giou限制在区间[-1.0, 1.0]内
            giou = torch.clamp(giou, -1.0, 1.0)
            loss = 1 - giou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1.0, eps=1e-10, reduction="none"):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def __call__(self, pbox, gbox, iou_weight=1.0, loc_reweight=None):
        # x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        # x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
        x1, y1, x2, y2 = torch.split(pbox, split_size_or_sections=1, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, split_size_or_sections=1, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = bbox_overlaps(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == "none":
            loss = giou
        elif self.reduction == "sum":
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight


@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


@weighted_loss
def diou_loss(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw ** 2 + ch ** 2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


@weighted_loss
def ciou_loss(pred, target, eps=1e-7):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw ** 2 + ch ** 2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    factor = 4 / math.pi ** 2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    # CIoU
    cious = ious - (rho2 / c2 + v ** 2 / (1 - ious + v))
    loss = 1 - cious
    return loss


class MyGIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(MyGIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


class MyDIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(MyDIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * diou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss


class MyCIoULoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="mean", loss_weight=1.0):
        super(MyCIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * ciou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
        return loss
