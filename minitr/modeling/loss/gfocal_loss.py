import torch.nn as nn
import torch.nn.functional as F
import torch
from .loss import weight_reduce_loss
from yolov7.utils.misc import weighted_loss


@weighted_loss
def quality_focal_loss(pred, target, beta=2.0):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert (
        len(target) == 2
    ), """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction="none"
    ) * scale_factor.pow(beta)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.size(1)
    pos = torch.nonzero((label >= 0) & (label < bg_class_ind), as_tuple=False).squeeze(
        1
    )
    pos_label = label[pos].long()
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction="none"
    ) * scale_factor.abs().pow(beta)

    loss = loss.sum(dim=1, keepdim=False)
    return loss


def distribution_focal_loss(
    pred, label, weight=None, reduction="mean", avg_factor=None
):
    disl = label.long()
    disr = disl + 1

    wl = disr.float() - label
    wr = label - disl.float()

    loss = (
        F.cross_entropy(pred, disl, reduction="none") * wl
        + F.cross_entropy(pred, disr, reduction="none") * wr
    )
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class QualityFocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, beta=2.0, reduction="mean", loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, "Only sigmoid in QFL supported now."
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor,
            )
        else:
            raise NotImplementedError
        return loss_cls


class DistributionFocalLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls
