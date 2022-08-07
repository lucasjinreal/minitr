from minitr.modeling.loss.setcriterion import FocalLossSetCriterion
from torch import nn
from minitr.utils.initializer import bias_init_with_prob
import torch
from alfred import device
from minitr.utils.boxes import box_xyxy_to_cxcywh, convert_coco_poly_to_mask


class DETRHead(nn.Module):

    """
    Implement DETRHead, do GT matching and return the losses
    """

    def __init__(self, cfg, hidden_dim, num_queries, num_classes=80) -> None:
        super().__init__()

        self.set_loss = FocalLossSetCriterion()

        cls_weight = cfg.MODEL.DETR.CLS_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        giou_weight = cfg.MODEL.DETR.IOU_WEIGHT

        # building criterion
        matcher = HungarianMatcherD2go(
            cost_class=cls_weight,
            cost_bbox=l1_weight,
            cost_giou=giou_weight,
            use_focal_loss=self.use_focal_loss,
        )
        weight_dict = {"loss_ce": cls_weight, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        nn.init.constant_(self.class_embed.bias, bias_init_with_prob(0.01))
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.aux_loss = aux_loss
        self.set_loss.to(self.device)

    def forward(self, x, targets=None):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print(samples)
        # print(samples[1])
        # h_w = torch.stack([torch.stack([inst['size'] for inst in samples[1]])[:, 1],
        #                    torch.stack([inst['size'] for inst in samples[1]])[:, 0]], dim=-1)
        # h_w = h_w.unsqueeze(0)
        h_w = torch.stack(
            [
                torch.stack([torch.tensor(inst) for inst in samples.image_sizes])[:, 1],
                torch.stack([torch.tensor(inst) for inst in samples.image_sizes])[:, 0],
            ],
            dim=-1,
        )
        # print(h_w)
        # h_w = torch.tensor(samples.image_sizes).to(device)
        h_w = h_w.unsqueeze(0).to(device)

        src, mask = features[-1].decompose()
        # print(f'{src.shape} {mask.shape}')
        assert mask is not None
        hs, points = self.transformer(
            self.input_proj(src), mask, self.query_embed.weight, pos[-1], h_w
        )
        num_decoder = hs.shape[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)

        points = points.unsqueeze(0).repeat(num_decoder, 1, 1, 1)

        outputs_coord[..., :2] = outputs_coord[..., :2] + points
        outputs_coord = outputs_coord.sigmoid()
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.training:
            loss = self.get_loss(out, targets)
            return loss
        else:
            return out

    def get_loss(self, output, gt_instances):
        # targets: List[Dict[str, torch.Tensor]]. Keys
        # "labels": [NUM_BOX,]
        # "boxes": [NUM_BOX, 4]
        targets = self.prepare_targets(gt_instances)
        loss_dict = self.criterion(output, targets)
        weight_dict = self.criterion.weight_dict
        for k in loss_dict.keys():
            if not loss_dict[k].requires_grad:
                loss_dict[k] = loss_dict[k].new_tensor(0)
            if k in weight_dict:
                loss_dict[k] *= weight_dict[k]
        return loss_dict

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=self.device
            )
            gt_classes = targets_per_image.gt_classes  # shape (NUM_BOX,)
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)  # shape (NUM_BOX, 4)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, "gt_masks"):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({"masks": gt_masks})
        return new_targets
