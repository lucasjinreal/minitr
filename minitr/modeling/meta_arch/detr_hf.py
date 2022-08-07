"""
Simpified DETR
construction with HuggingFace design

"""
from torch import nn
import torch

import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from ..transcoders.decoder_detr import DetrDecoderLayer
from ..transcoders.encoder_detr import DetrEncoder, DetrEncoderLayer

from alfred import logger


@META_ARCH_REGISTRY.register()
class DETRSimple(nn.Module):
    """
    Simplier verision DETR in Detectron2
    borrowed some design from HuggingFace
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON

        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        cls_weight = cfg.MODEL.DETR.CLS_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT
        centered_position_encoding = cfg.MODEL.DETR.CENTERED_POSITION_ENCODIND
        num_feature_levels = cfg.MODEL.DETR.NUM_FEATURE_LEVELS

        N_steps = hidden_dim // 2
        if "resnet" in cfg.MODEL.BACKBONE.NAME.lower():
            d2_backbone = ResNetMaskedBackbone(cfg)
        elif "fbnet" in cfg.MODEL.BACKBONE.NAME.lower():
            d2_backbone = FBNetMaskedBackbone(cfg)
        elif cfg.MODEL.BACKBONE.SIMPLE:
            d2_backbone = SimpleSingleStageBackbone(cfg)
        else:
            raise NotImplementedError

        backbone = Joiner(
            d2_backbone,
            PositionEmbeddingSine(
                N_steps, normalize=True, centered=centered_position_encoding
            ),
        )
        backbone.num_channels = d2_backbone.num_channels
        self.use_focal_loss = cfg.MODEL.DETR.USE_FOCAL_LOSS

        if cfg.MODEL.DETR.ATTENTION_TYPE == "deformable":
            logger.info("Deformable not supported now.")
        elif cfg.MODEL.DETR.ATTENTION_TYPE == "DETR":
            transformer = DetrTransformer(
                d_model=hidden_dim,
                dropout=dropout,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                num_encoder_layers=enc_layers,
                num_decoder_layers=dec_layers,
                normalize_before=pre_norm,
                return_intermediate_dec=deep_supervision,
            )
            self.detr = DETR(
                backbone,
                transformer,
                num_classes=self.num_classes,
                num_queries=num_queries,
                aux_loss=deep_supervision,
                use_focal_loss=self.use_focal_loss,
            )
        else:
            logger.error(f"attention type: {cfg.MODEL.DETR.ATTENTION_TYPE} not support")
            exit(1)

        if self.mask_on:
            frozen_weights = cfg.MODEL.DETR.FROZEN_WEIGHTS
            if frozen_weights != "":
                print("LOAD pre-trained weights")
                weight = torch.load(
                    frozen_weights, map_location=lambda storage, loc: storage
                )["model"]
                new_weight = {}
                for k, v in weight.items():
                    if "detr." in k:
                        new_weight[k.replace("detr.", "")] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.detr.load_state_dict(new_weight)
                del new_weight
            self.detr = DETRsegm(self.detr, freeze_detr=(frozen_weights != ""))
            self.seg_postprocess = PostProcessSegm

        self.detr.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_input(self, x):
        # x = x.permute(0, 3, 1, 2)
        # x = F.interpolate(x, size=(640, 640))
        # x = F.interpolate(x, size=(512, 960))
        """
        x is N, CHW aleady permuted
        """
        x = [self.normalizer(i) for i in x]
        return x

    def forward(self, batched_inputs):
        if torch.onnx.is_in_onnx_export():
            logger.info("[WARN] exporting onnx...")
            assert isinstance(batched_inputs, (list, torch.Tensor)) or isinstance(
                batched_inputs, list
            ), "onnx export, batched_inputs only needs image tensor or list of tensors"
            images = self.preprocess_input(batched_inputs)
            # batched_inputs = batched_inputs.permute(0, 3, 1, 2)
            # image_ori_sizes = [batched_inputs.shape[1:3]]
        else:
            images = self.preprocess_image(batched_inputs)
        output = self.detr(images)

        if self.training:
            pass
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            results = self.inference(
                box_cls, box_pred, mask_pred, images.image_sizes
            )
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        if self.use_focal_loss:
            prob = box_cls.sigmoid()
            # TODO make top-100 as an option for non-focal-loss as well
            scores, topk_indexes = torch.topk(
                prob.view(box_cls.shape[0], -1), 100, dim=1
            )
            topk_boxes = topk_indexes // box_cls.shape[2]
            labels = topk_indexes % box_cls.shape[2]
        else:
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (
            scores_per_image,
            labels_per_image,
            box_pred_per_image,
            image_size,
        ) in enumerate(zip(scores, labels, box_pred, image_sizes)):
            result = Instances(image_size)
            boxes = box_cxcywh_to_xyxy(box_pred_per_image)
            if self.use_focal_loss:
                boxes = torch.gather(boxes, 0, topk_boxes[i].unsqueeze(-1).repeat(1, 4))

            result.pred_boxes = Boxes(boxes)
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(
                    mask_pred[i].unsqueeze(0),
                    size=image_size,
                    mode="bilinear",
                    align_corners=False,
                )
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(
                    result.pred_boxes.tensor.cpu(), 32
                )
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images


class DetrModel(nn.Module):
    def __init__(self, config: DetrConfig):
        super().__init__(config)

        # Create backbone + positional encoding
        backbone = DetrTimmConvEncoder(
            config.backbone, config.dilation, config.use_pretrained_backbone
        )
        position_embeddings = build_position_encoding(config)
        self.backbone = DetrConvModel(backbone, position_embeddings)

        # Create projection layer
        self.input_projection = nn.Conv2d(
            backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1
        )

        self.query_position_embeddings = nn.Embedding(
            config.num_queries, config.d_model
        )

        self.encoder = DetrEncoder(config)
        self.decoder = DetrEncoderLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def freeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    def unfreeze_backbone(self):
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    def forward(
        self,
        pixel_values,
        pixel_mask=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import DetrFeatureExtractor, DetrModel
        >>> from PIL import Image
        >>> import requests
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        >>> model = DetrModel.from_pretrained("facebook/detr-resnet-50")
        >>> inputs = feature_extractor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 100, 256]
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device

        if pixel_mask is None:
            pixel_mask = torch.ones(((batch_size, height, width)), device=device)

        # First, sent pixel_values + pixel_mask through Backbone to obtain the features
        # pixel_values should be of shape (batch_size, num_channels, height, width)
        # pixel_mask should be of shape (batch_size, height, width)
        features, position_embeddings_list = self.backbone(pixel_values, pixel_mask)

        # get final feature map and downsampled mask
        feature_map, mask = features[-1]

        if mask is None:
            raise ValueError("Backbone does not return downsampled pixel mask")

        # Second, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        projected_feature_map = self.input_projection(feature_map)

        # Third, flatten the feature map + position embeddings of shape NxCxHxW to NxCxHW, and permute it to NxHWxC
        # In other words, turn their shape into (batch_size, sequence_length, hidden_size)
        flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
        position_embeddings = position_embeddings_list[-1].flatten(2).permute(0, 2, 1)

        flattened_mask = mask.flatten(1)

        # Fourth, sent flattened_features + flattened_mask + position embeddings through encoder
        # flattened_features is a Tensor of shape (batch_size, heigth*width, hidden_size)
        # flattened_mask is a Tensor of shape (batch_size, heigth*width)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # Fifth, sent query embeddings + position embeddings through the decoder (which is conditioned on the encoder output)
        query_position_embeddings = self.query_position_embeddings.weight.unsqueeze(
            0
        ).repeat(batch_size, 1, 1)
        queries = torch.zeros_like(query_position_embeddings)

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            inputs_embeds=queries,
            attention_mask=None,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=flattened_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return DetrModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
        )
