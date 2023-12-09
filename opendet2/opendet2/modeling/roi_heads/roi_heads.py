# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import inspect
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.structures.boxes import pairwise_intersection
from detectron2.layers import ShapeSpec, cat, cross_entropy
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import (
    ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads, add_ground_truth_to_proposals, select_foreground_proposals,
    select_proposals_with_visible_keypoints)
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats, FastRCNNOutputLayers
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn

from .fast_rcnn import build_roi_box_output_layers
from ..vmf import vMFLogPartition

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class OpenSetStandardROIHeads(StandardROIHeads):

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]) -> List[Instances]:
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            # NOTE: add iou of each proposal
            ious, _ = match_quality_matrix.max(dim=0)
            proposals_per_image.iou = ious[sampled_idxs]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels,
                           height=pooler_resolution, width=pooler_resolution)
        )
        # register output layers
        box_predictor = build_roi_box_output_layers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }


@ROI_HEADS_REGISTRY.register()
class DropoutStandardROIHeads(OpenSetStandardROIHeads):
    @configurable
    def __init__(self, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        # num of sampling
        self.num_sample = 30

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], targets=None):

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        # if testing, we run multiple inference for dropout sampling
        if self.training:
            predictions = self.box_predictor(box_features)
        else:
            predictions = [self.box_predictor(
                box_features, testing=True) for _ in range(self.num_sample)]

        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(
                            pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(
                predictions, proposals)
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class SIRENOpenSetStandardROIHeads(ROIHeads):
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        # TODO define projection layer
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        # copy from siren
        self.iterations = self.cfg.SOLVER.MAX_ITER
        self.vmf_loss_weight = self.cfg.SIREN.LOSS_WEIGHT
        # TODO use config to set self.output_dir
        self.output_dir = "./saved_mean_kappa"

        # TODO add projection_dim to config file
        # self.projection_dim = self.cfg.SIREN.PROJECTION_DIM # sometimes 32 is better. TBD
        self.projection_dim = 64

        self.projection_head = nn.Sequential(
            nn.Linear(1024, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        )

        # siren component
        self.prototypes = torch.zeros(
            (self.num_classes, self.projection_dim)
        ).cuda()  # prototypes is a simple n X 64 tensor.
        self.learnable_kappa = nn.Linear(self.num_classes, 1, bias=False).cuda()
        nn.init.constant(self.learnable_kappa.weight, 10)

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            # NOTE: add iou of each proposal
            ious, _ = match_quality_matrix.max(dim=0)
            proposals_per_image.iou = ious[sampled_idxs]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for trg_name, trg_value in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        # Add this to obtain config fot self.box_predictor.losses() func
        cls.cfg = cfg
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        # register output layers
        box_predictor = build_roi_box_output_layers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    # to replace the StandardROIHeads's _forward_box function
    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        iteration: Optional[int] = None,
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)  # reg_feat, cls_feat
        predictions = self.box_predictor(box_features)  # scores, proposal, mlp_feat
        # added from siren, use (reg_feat + cls_feat) as input of the projection head
        projections = self.projection_head(
            # torch.add(box_features[0], box_features[1])
            box_features[1]
        )  # (reg_feat + cls_feat)

        del box_features

        if self.training:
            # losses = self.box_predictor.losses(predictions, proposals, projections)  # in siren, it just move loss func below

            # proposals is modified in-place below, so losses must be computed first.
            scores, proposal_deltas, _ = predictions

            # parse classification outputs

            gt_classes = (
                cat([p.gt_classes for p in proposals], dim=0)
                if len(proposals)
                else torch.empty(0)
            )
            gt_classes_filtered = gt_classes[gt_classes != self.num_classes].cuda()
            projections_filtered = projections[gt_classes != self.num_classes].cuda()
            _log_classification_stats(scores, gt_classes)

            # self.sample_number = 10
            # print(iteration)
            # parse box regression outputs
            if len(proposals):
                proposal_boxes = cat(
                    [p.proposal_boxes.tensor for p in proposals], dim=0
                )  # Nx4
                assert (
                    not proposal_boxes.requires_grad
                ), "Proposals should not require gradients!"
                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = cat(
                    [
                        (p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor
                        for p in proposals
                    ],
                    dim=0,
                )

                for class_i, projection in zip(
                    gt_classes_filtered, projections_filtered
                ):
                    self.prototypes.data[class_i] = F.normalize(
                        0.05 * F.normalize(projection, p=2, dim=-1)
                        + 0.95 * self.prototypes.data[class_i],
                        p=2,
                        dim=-1,
                    )  # EMA updating

                cosine_logits = F.cosine_similarity(  # calculate cosine similarity between self.prototype and projections filtered.
                    self.prototypes.data.detach()
                    .unsqueeze(0)
                    .repeat(len(projections_filtered), 1, 1),
                    projections_filtered.unsqueeze(1).repeat(1, self.num_classes, 1),
                    2,
                )

                weight_before_exp = vMFLogPartition.apply(
                    self.projection_dim,  # apply can be seen as forward() of the class
                    F.relu(self.learnable_kappa.weight.view(1, -1)),
                )
                weight_before_exp = weight_before_exp.exp()
                cosine_similarity_loss = self.weighted_vmf_loss(
                    cosine_logits * F.relu(self.learnable_kappa.weight.view(1, -1)),
                    weight_before_exp,
                    gt_classes_filtered,
                )

                if iteration == self.iterations - 1:
                    np.save(
                        self.output_dir + "/proto.npy",
                        self.prototypes.cpu().data.numpy(),
                    )  # μ, prototype
                    np.save(
                        self.output_dir + "/kappa.npy",
                        self.learnable_kappa.weight.cpu().data.numpy(),
                    )  # κ, kappa

                del weight_before_exp

            losses = {
                "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                "loss_center": cosine_similarity_loss * self.vmf_loss_weight,
                "loss_box_reg": self.box_predictor.box_reg_loss(
                    proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                ),
            }
            # add ic loss and up loss from opendet
            losses = self.box_predictor.losses(
                predictions, proposals, losses, projections_filtered
            )
            losses = {
                k: v * self.box_predictor.loss_weight.get(k, 1.0)
                for k, v in losses.items()
            }

            # below is the traditional
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes for x in instances
            ]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes for x in instances
            ]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def weighted_vmf_loss(self, pred, weight_before_exp, target):
        center_adaptive_weight = weight_before_exp.view(1, -1)
        pred = (
            center_adaptive_weight
            * pred.exp()
            / ((center_adaptive_weight * pred.exp()).sum(-1)).unsqueeze(-1)
        )
        loss = -(pred[range(target.shape[0]), target] + 1e-6).log().mean()

        return loss
    
    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        storage = get_event_storage()
        iteration = storage.iter
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            # iteration is added for output the final siren prototype and κ
            losses = self._forward_box(features, proposals, iteration)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

@ROI_HEADS_REGISTRY.register()
class ROIHeadsLogisticGMMNew(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.
        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

        self.sample_number = self.cfg.VOS.SAMPLE_NUMBER
        self.start_iter = self.cfg.VOS.STARTING_ITER
        # print(self.sample_number, self.start_iter)

        self.logistic_regression = torch.nn.Linear(1, 2)
        self.logistic_regression.cuda()
        # torch.nn.init.xavier_normal_(self.logistic_regression.weight)

        self.select = 1
        self.sample_from = 10000
        self.loss_weight = 0.1
        self.weight_energy = torch.nn.Linear(self.num_classes, 1).cuda()
        torch.nn.init.uniform_(self.weight_energy.weight)
        self.data_dict = torch.zeros(self.num_classes, self.sample_number, 1024).cuda()
        self.number_dict = {}
        self.eye_matrix = torch.eye(1024, device='cuda')
        self.trajectory = torch.zeros((self.num_classes, 900, 3)).cuda()
        for i in range(self.num_classes):
            self.number_dict[i] = 0
        self.cos = torch.nn.MSELoss()  #
        self.complete_scores = nn.Linear(in_features = 1024, out_features = 1)
        # self.similarity_feat = nn.Linear(in_features = 1024, out_features = 256)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        cls.cfg = cfg
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        iteration: int,
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, iteration)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, iteration)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def log_sum_exp(self, value, dim=None, keepdim=False):
        """Numerically stable implementation of the operation
        value.exp().sum(dim, keepdim).log()
        """
        import math
        # TODO: torch.max(value, dim=None) threw an error at time of writing
        if dim is not None:
            m, _ = torch.max(value, dim=dim, keepdim=True)
            value0 = value - m
            if keepdim is False:
                m = m.squeeze(dim)
            return m + torch.log(torch.sum(
                F.relu(self.weight_energy.weight) * torch.exp(value0),dim=dim, keepdim=keepdim))
        else:
            m = torch.max(value)
            sum_exp = torch.sum(torch.exp(value - m))
            # if isinstance(sum_exp, Number):
            #     return m + math.log(sum_exp)
            # else:
            return m + torch.log(sum_exp)

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], iteration: int):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        predictions = self.box_predictor(box_features)
        # del box_features
        # breakpoint()

        if self.training:
            # losses = self.box_predictor.losses(predictions, proposals)

            scores, proposal_deltas = predictions

            # parse classification outputs

            gt_classes = (
                cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
            )

            _log_classification_stats(scores, gt_classes)
            # self.sample_number = 10
            # print(iteration)
            # parse box regression outputs
            if len(proposals):
                proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
                assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = cat(
                    [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                    dim=0,
                )

                sum_temp = 0
                for index in range(self.num_classes):
                    sum_temp += self.number_dict[index]
                # print(iteration)
                lr_reg_loss = torch.zeros(1).cuda()
                if sum_temp == self.num_classes * self.sample_number and iteration < self.start_iter:
                    selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                    indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                    gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                    # maintaining an ID data queue for each class.
                    for index in indices_numpy:
                        dict_key = gt_classes_numpy[index]
                        self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                              box_features[index].detach().view(1, -1)), 0)
                elif sum_temp == self.num_classes * self.sample_number and iteration >= self.start_iter:
                    selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                    indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                    gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                    # maintaining an ID data queue for each class.
                    for index in indices_numpy:
                        dict_key = gt_classes_numpy[index]
                        self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                              box_features[index].detach().view(1, -1)), 0)
                    # the covariance finder needs the data to be centered.
                    for index in range(self.num_classes):
                        if index == 0:
                            X = self.data_dict[index] - self.data_dict[index].mean(0)
                            mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
                        else:
                            X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                            mean_embed_id = torch.cat((mean_embed_id,
                                                       self.data_dict[index].mean(0).view(1, -1)), 0)

                    # add the variance.
                    temp_precision = torch.mm(X.t(), X) / len(X)
                    # for stable training.
                    temp_precision += 0.0001 * self.eye_matrix


                    for index in range(self.num_classes):
                        new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                            mean_embed_id[index], covariance_matrix=temp_precision)
                        negative_samples = new_dis.rsample((self.sample_from,))
                        prob_density = new_dis.log_prob(negative_samples)

                        # keep the data in the low density area.
                        cur_samples, index_prob = torch.topk(- prob_density, self.select)
                        if index == 0:
                            ood_samples = negative_samples[index_prob]
                        else:
                            ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                        del new_dis
                        del negative_samples


                    if len(ood_samples) != 0:
                        # add some gaussian noise
                        # ood_samples = self.noise(ood_samples)
                        # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                        energy_score_for_fg = self.log_sum_exp(predictions[0][selected_fg_samples][:, :-1], 1)
                        predictions_ood = self.box_predictor(ood_samples)
                        # # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                        energy_score_for_bg = self.log_sum_exp(predictions_ood[0][:, :-1], 1)

                        input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                        labels_for_lr = torch.cat((torch.ones(len(selected_fg_samples)).cuda(),
                                                   torch.zeros(len(ood_samples)).cuda()), -1)
                        if False:
                            output = self.logistic_regression(input_for_lr.view(-1, 1))
                            lr_reg_loss = F.binary_cross_entropy_with_logits(
                                output.view(-1), labels_for_lr)
                            print(torch.cat((
                                F.sigmoid(output[:len(selected_fg_samples)]).view(-1, 1),
                                          torch.ones(len(selected_fg_samples), 1).cuda()), 1))
                            print(torch.cat((
                                F.sigmoid(output[len(selected_fg_samples):]).view(-1, 1),
                                torch.zeros(len(ood_samples), 1).cuda()), 1))
                        else:
                            weights_fg_bg = torch.Tensor([len(selected_fg_samples) / float(len(input_for_lr)),
                                                         len(ood_samples) / float(len(input_for_lr))]).cuda()
                            criterion = torch.nn.CrossEntropyLoss()#weight=weights_fg_bg)
                            output = self.logistic_regression(input_for_lr.view(-1, 1))
                            lr_reg_loss = criterion(output, labels_for_lr.long())

                    del ood_samples

                else:
                    selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                    indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                    gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                    for index in indices_numpy:
                        dict_key = gt_classes_numpy[index]
                        if self.number_dict[dict_key] < self.sample_number:
                            self.data_dict[dict_key][self.number_dict[dict_key]] = box_features[index].detach()
                            self.number_dict[dict_key] += 1
                # create a dummy in order to have all weights to get involved in for a loss.
                loss_dummy = self.cos(self.logistic_regression(torch.zeros(1).cuda()), self.logistic_regression.bias)
                loss_dummy1 = self.cos(self.weight_energy(torch.zeros(self.num_classes).cuda()), self.weight_energy.bias)
                del box_features
                # print(self.number_dict)

            else:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

            if sum_temp == self.num_classes * self.sample_number:
                losses = {
                    "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                    "lr_reg_loss": self.loss_weight * lr_reg_loss,
                    "loss_dummy": loss_dummy,
                    "loss_dummy1": loss_dummy1,
                    "loss_box_reg": self.box_predictor.box_reg_loss(
                        proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                    ),
                }
            else:
                losses = {
                    "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                    "lr_reg_loss":torch.zeros(1).cuda(),
                    "loss_dummy": loss_dummy,
                    "loss_dummy1": loss_dummy1,
                    "loss_box_reg": self.box_predictor.box_reg_loss(
                        proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                    ),
                }
            losses =  {k: v * self.box_predictor.loss_weight.get(k, 1.0) for k, v in losses.items()}

            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            # https://github.com/pytorch/pytorch/issues/49728
            if self.training:
                return {}
            else:
                return instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the keypoint prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            # https://github.com/pytorch/pytorch/issues/49728
            if self.training:
                return {}
            else:
                return instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = dict([(f, features[f]) for f in self.keypoint_in_features])
        return self.keypoint_head(features, instances)

def LIoU(boxes1: Boxes, boxes2: Boxes, denominator=0):
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] if denominator==0 else area2),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

@ROI_HEADS_REGISTRY.register()
class OpensetROIHeadsLogisticGMMNewLimitFPEnergy(ROIHeadsLogisticGMMNew):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        storage = get_event_storage()
        iteration = storage.iter
        if self.training:
            assert targets, "'targets' argument is required during training"
            if iteration >= self.start_iter:
                goc_samples = self.sample_localboxes_and_completeboxes_t2(proposals, targets)
            else:
                goc_samples = None
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, iteration, goc_samples)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, iteration)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
            
    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).

        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # breakpoint()
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )
            # breakpoint()
            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            # NOTE: Opendet: add iou of each proposal
            ious, _ = match_quality_matrix.max(dim=0)
            proposals_per_image.iou = ious[sampled_idxs]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.
            # print(proposals_per_image)
            # breakpoint()
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
    
    
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels,
                           height=pooler_resolution, width=pooler_resolution)
        )
        # register output layers
        box_predictor = build_roi_box_output_layers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }


    def export_pos_box(self, proposal_boxes, iou_with_gt, match_gtid, gt_num, e2):
        pos_x_proposals = []
        pos_y_proposals = []
        pos_bool_list = []
        pos_index = torch.where(iou_with_gt >= e2)
        pos_proposal_boxes = proposal_boxes[pos_index]
        pos_iou_with_gt = iou_with_gt[pos_index]
        pos_match_gtid = match_gtid[pos_index]
        instances_ids = []
        for gt_id in range(gt_num):
            propid_matched_thisgt = torch.where(pos_match_gtid == gt_id) 
            x = np.random.choice(len(propid_matched_thisgt[0]), int(len(propid_matched_thisgt[0]) / 2))
            y = np.random.choice(len(propid_matched_thisgt[0]), int(len(propid_matched_thisgt[0]) / 2))
            np.random.shuffle(y)
            proposals_idx = propid_matched_thisgt[0][x]
            proposals_idy = propid_matched_thisgt[0][y]
            proposals_x = pos_proposal_boxes[proposals_idx]
            proposals_y = pos_proposal_boxes[proposals_idy]
            iou_x = pos_iou_with_gt[proposals_idx]
            iou_y = pos_iou_with_gt[proposals_idy]
            # sort
            sort_x = iou_x.sort()[1]
            sort_y = iou_y.sort()[1]
            proposals_x = proposals_x[sort_x]
            proposals_y = proposals_y[sort_y]
            iou_x = iou_x[sort_x]
            iou_y = iou_y[sort_y]

            bool_ioux_bigger_louy = (iou_x > iou_y).int()
            # remove = 
            keep = iou_x != iou_y
            pos_x_proposals.append(proposals_x[keep])
            pos_y_proposals.append(proposals_y[keep])
            pos_bool_list.append(bool_ioux_bigger_louy[keep])
            # instances_ids = instances_ids + [gt_id] * keep.sum().item()
            pass
        return pos_x_proposals, pos_y_proposals, pos_bool_list#, instances_ids

    def export_neg_box(self, gt_boxes, proposal_boxes, iou_with_gt, match_gtid, gt_num, e2):
        p1 = 0.2
        g1 = 0.2
        neg_index = torch.where(iou_with_gt < e2)  # 0 < iou < 0.5
        neg_proposal_boxes = proposal_boxes[neg_index]
        neg_iou_with_gt = iou_with_gt[neg_index]
        neg_match_gtid = match_gtid[neg_index]
        gt_ratio = LIoU(gt_boxes, neg_proposal_boxes, 0)  # inter / gt area
        pro_ratio = LIoU(gt_boxes, neg_proposal_boxes, 1)  # inter / proposal area
        neg_proposals = []
        for gt_id in range(gt_num):
            propid_matched_thisgt = torch.where(neg_match_gtid == gt_id) 
            gt_ratio_thisgt = gt_ratio[gt_id, propid_matched_thisgt[0]] 
            include_more_gt_index = torch.where(gt_ratio_thisgt >= g1)
            neg_proposals.append(neg_proposal_boxes[propid_matched_thisgt][include_more_gt_index])

            pro_ratio_thisgt = pro_ratio[gt_id, propid_matched_thisgt[0]]
            include_more_pro_index = torch.where(pro_ratio_thisgt >= p1)
            neg_proposals.append(neg_proposal_boxes[propid_matched_thisgt][include_more_pro_index])
        return neg_proposals

    def sample_localboxes_and_completeboxes_t2(self, proposals, targets):
        proposals_batch_pos_x = []
        proposals_batch_pos_y = []
        bool_pos_batch = []
        proposals_batch_neg = []
        # batch_instances_ids = []# gt'id that proposals belong to 
        e1 = 0.0#0.0
        e2 = 0.5#0.5
        for i, img in enumerate(proposals):
            proposal_boxes = img.proposal_boxes
            gt_boxes = targets[i].gt_boxes
            gt_num = targets[i].gt_classes.shape[0]
            iou_matrix = pairwise_iou(gt_boxes, proposal_boxes) # gt * pro
            
            iou_with_gt, match_gtid = iou_matrix.max(0)
            potential_index = torch.where(iou_with_gt > e1) 
            proposal_boxes = proposal_boxes[potential_index] # 更新proposals, iou, match_gt_id
            iou_with_gt = iou_with_gt[potential_index]
            match_gtid = match_gtid[potential_index]

            pos_x_proposals, pos_y_proposals, pos_bool_list = self.export_pos_box(proposal_boxes, iou_with_gt, match_gtid, gt_num, e2)
            neg_proposals = self.export_neg_box(gt_boxes, proposal_boxes, iou_with_gt, match_gtid, gt_num, e2)
            
            
            proposals_batch_pos_x.append(Boxes.cat(pos_x_proposals)) 
            proposals_batch_pos_y.append(Boxes.cat(pos_y_proposals))    
            bool_pos_batch.append(torch.cat(pos_bool_list))
            
            neg_proposals = Boxes.cat(neg_proposals)
            index = np.random.choice(len(neg_proposals), min(500, len(neg_proposals)))
            index = torch.from_numpy(index).cuda()    
            proposals_batch_neg.append(neg_proposals[index])
            
            # batch_instances_ids.append(torch.tensor(instances_ids, dtype=torch.int))

        return proposals_batch_pos_x, proposals_batch_pos_y, bool_pos_batch, proposals_batch_neg
    
    def extract_feature(self, features, proposals_batch):
        box_features = self.box_pooler(features, proposals_batch)
        box_features = self.box_head(box_features)
        return box_features

    def loss_local_complete_t2(self, goc_samples, features):
        proposals_batch_x, proposals_batch_y, bool_batch, neg_batch = goc_samples
        # 1: x > y, 0: y > x
        bool_batch = torch.cat(bool_batch)
        box_features_x = self.extract_feature(features, proposals_batch_x)
        complete_scores_x = self.complete_scores(box_features_x[0])

        box_features_y = self.extract_feature(features, proposals_batch_y)
        complete_scores_y = self.complete_scores(box_features_y[0])

        # contrast_loss = self.contrast_loss(box_features_x, box_features_y, batch_instances_ids)

        loss_pos1 = (1 - bool_batch) * F.relu(complete_scores_x + 1e-2 - complete_scores_y) + \
            (bool_batch) * F.relu(complete_scores_y + 1e-2 - complete_scores_x)
        MSELoss = nn.MSELoss()
        loss_pos2 = MSELoss(complete_scores_x, torch.ones_like(complete_scores_x))#F.relu(-1 * torch.log(complete_scores_x + 0.0001))
        loss_pos3 = MSELoss(complete_scores_y, torch.ones_like(complete_scores_y))#F.relu(-1 * torch.log(complete_scores_y + 0.0001))

        box_features_neg = self.extract_feature(features, neg_batch)
        complete_scores_neg = self.complete_scores(box_features_neg[0])
        minscores = 0.5 #if tmp.shape[0] == 0 else tmp.min().item()
        loss_neg = F.relu(complete_scores_neg + 1e-3 - minscores)#MSELoss(complete_scores_neg, torch.zeros_like(complete_scores_neg))#F.relu(complete_scores_neg)
        
        pos_weight = 1 if bool_batch.shape[0] > 0 else 0
        neg_weight = 1 if complete_scores_neg.shape[0] > 0 else 0
        loss = pos_weight * (loss_pos1.mean() + loss_pos2.mean() + loss_pos3.mean()) + loss_neg.mean() * neg_weight
        # print("shape: {}  {}   ".format(bool_batch.shape[0], complete_scores_neg.shape[0]))
        print("minscores:{:.4f} | loss_pos1.mean:{:.4f} | loss_pos2.mean():{:.4f} | loss_pos3.mean:{:.4f} | loss_neg.mean:{:.4f}".format(minscores, loss_pos1.mean(), loss_pos2.mean(), loss_pos3.mean(), loss_neg.mean()))
        return loss


    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], iteration: int,
                        goc_samples=None):
            """
            Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
                the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
            Args:
                features (dict[str, Tensor]): mapping from feature map names to tensor.
                    Same as in :meth:`ROIHeads.forward`.
                proposals (list[Instances]): the per-image object proposals with
                    their matching ground truth.
                    Each has fields "proposal_boxes", and "objectness_logits",
                    "gt_classes", "gt_boxes".
            Returns:
                In training, a dict of losses.
                In inference, a list of `Instances`, the predicted instances.
            """

            features = [features[f] for f in self.box_in_features]
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.box_head(box_features)
            
            predictions = self.box_predictor(box_features)
            # del box_features
            # breakpoint()

            if self.training:
                # losses = self.box_predictor.losses(predictions, proposals)

                scores, proposal_deltas, _ = predictions

                # parse classification outputs

                gt_classes = (
                    cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
                )

                _log_classification_stats(scores, gt_classes)
                # self.sample_number = 10
                # print(iteration)
                # parse box regression outputs
                if len(proposals):
                    proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
                    assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
                    # If "gt_boxes" does not exist, the proposals must be all negative and
                    # should not be included in regression loss computation.
                    # Here we just use proposal_boxes as an arbitrary placeholder because its
                    # value won't be used in self.box_reg_loss().
                    gt_boxes = cat(
                        [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                        dim=0,
                    )

                    sum_temp = 0
                    for index in range(self.num_classes):
                        sum_temp += self.number_dict[index]
                    # print(iteration)
                    lr_reg_loss = torch.zeros(1).cuda()
                    if sum_temp == self.num_classes * self.sample_number and iteration < self.start_iter:
                        selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                        indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                        gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                        # maintaining an ID data queue for each class.
                        for index in indices_numpy:
                            dict_key = gt_classes_numpy[index]
                            self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                                box_features[0][index].detach().view(1, -1)), 0)
                    elif sum_temp == self.num_classes * self.sample_number and iteration >= self.start_iter:
                        selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                        indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                        gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                        # maintaining an ID data queue for each class.
                        for index in indices_numpy:
                            dict_key = gt_classes_numpy[index]
                            self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                                box_features[0][index].detach().view(1, -1)), 0)
                        # the covariance finder needs the data to be centered.
                        for index in range(self.num_classes):
                            if index == 0:
                                X = self.data_dict[index] - self.data_dict[index].mean(0)
                                mean_embed_id = self.data_dict[index].mean(0).view(1, -1)
                            else:
                                X = torch.cat((X, self.data_dict[index] - self.data_dict[index].mean(0)), 0)
                                mean_embed_id = torch.cat((mean_embed_id,
                                                        self.data_dict[index].mean(0).view(1, -1)), 0)

                        # add the variance.
                        temp_precision = torch.mm(X.t(), X) / len(X)
                        # for stable training.
                        temp_precision += 0.0001 * self.eye_matrix


                        for index in range(self.num_classes):
                            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                                mean_embed_id[index], covariance_matrix=temp_precision)
                            negative_samples = new_dis.rsample((self.sample_from,))
                            prob_density = new_dis.log_prob(negative_samples)

                            # keep the data in the low density area.
                            cur_samples, index_prob = torch.topk(- prob_density, self.select)
                            if index == 0:
                                ood_samples = negative_samples[index_prob]
                            else:
                                ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                            del new_dis
                            del negative_samples


                        if len(ood_samples) != 0:
                            # add non-object
                            selected_non_obj_samples = (gt_classes == predictions[0].shape[1] - 1).nonzero().view(-1)
                            energy_score_for_non_obj = self.log_sum_exp(predictions[0][selected_non_obj_samples][:, :-1], 1)
                            sample_index_non_obj, sample_weight_non_obj = self.sample_non_obj(energy_score_for_non_obj)
                            energy_score_for_non_obj = energy_score_for_non_obj[sample_index_non_obj]
                            non_obj_loss = F.relu(energy_score_for_non_obj).mean() * 0.5 #1e-3

                            # add some gaussian noise
                            # ood_samples = self.noise(ood_samples)
                            # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                            energy_score_for_fg = self.log_sum_exp(predictions[0][selected_fg_samples][:, :-1], 1)
                            predictions_ood = self.box_predictor(ood_samples)
                            # # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                            energy_score_for_bg = self.log_sum_exp(predictions_ood[0][:, :-1], 1)

                            input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                            labels_for_lr = torch.cat((torch.ones(len(selected_fg_samples)).cuda(),
                                                    torch.zeros(len(ood_samples)).cuda(),), -1)
                            # input_for_lr = torch.cat((energy_score_for_non_obj, energy_score_for_bg, energy_score_for_fg), -1) 
                            # labels_for_lr = torch.cat((torch.zeros(len(sample_index_non_obj)).cuda(),
                            #                         torch.ones(len(ood_samples)).cuda(),
                            #                         torch.ones(len(selected_fg_samples)).cuda() * 2), -1)
                            if False:
                                output = self.logistic_regression(input_for_lr.view(-1, 1))
                                lr_reg_loss = F.binary_cross_entropy_with_logits(
                                    output.view(-1), labels_for_lr)
                                print(torch.cat((
                                    F.sigmoid(output[:len(selected_fg_samples)]).view(-1, 1),
                                            torch.ones(len(selected_fg_samples), 1).cuda()), 1))
                                print(torch.cat((
                                    F.sigmoid(output[len(selected_fg_samples):]).view(-1, 1),
                                    torch.zeros(len(ood_samples), 1).cuda()), 1))
                            else:

                                weights_fg_bg = torch.Tensor([len(selected_fg_samples) / float(len(input_for_lr)),
                                                            len(ood_samples) / float(len(input_for_lr))]).cuda()
                                criterion = torch.nn.CrossEntropyLoss()#weight=weights_fg_bg)
                                output = self.logistic_regression(input_for_lr.view(-1, 1))
                                lr_reg_loss = criterion(output, labels_for_lr.long()) + non_obj_loss

                        del ood_samples

                    else:
                        selected_fg_samples = (gt_classes != predictions[0].shape[1] - 1).nonzero().view(-1)
                        indices_numpy = selected_fg_samples.cpu().numpy().astype(int)
                        gt_classes_numpy = gt_classes.cpu().numpy().astype(int)
                        for index in indices_numpy:
                            dict_key = gt_classes_numpy[index]
                            if self.number_dict[dict_key] < self.sample_number:
                                self.data_dict[dict_key][self.number_dict[dict_key]] = box_features[0][index].detach().unsqueeze(0)
                                self.number_dict[dict_key] += 1
                    # create a dummy in order to have all weights to get involved in for a loss.
                    loss_dummy = self.cos(self.logistic_regression(torch.zeros(1).cuda()), self.logistic_regression.bias)
                    loss_dummy1 = self.cos(self.weight_energy(torch.zeros(self.num_classes).cuda()), self.weight_energy.bias)
                    del box_features
                    # print(self.number_dict)

                else:
                    proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

                if iteration < self.start_iter:
                    loss_complete = torch.zeros(1).cuda()
                else:
                    loss_complete = self.loss_local_complete_t2(goc_samples, features)

                if sum_temp == self.num_classes * self.sample_number:
                    losses = {
                        "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                        "lr_reg_loss": self.loss_weight * lr_reg_loss,
                        "loss_dummy": loss_dummy,
                        "loss_dummy1": loss_dummy1,
                        "loss_box_reg": self.box_predictor.box_reg_loss(
                            proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                        ),
                        "loss_complete": loss_complete
                    }
                else:
                    losses = {
                        "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                        "lr_reg_loss":torch.zeros(1).cuda(),
                        "loss_dummy": loss_dummy,
                        "loss_dummy1": loss_dummy1,
                        "loss_box_reg": self.box_predictor.box_reg_loss(
                            proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                        ),
                        "loss_complete": loss_complete
                    }
                losses =  {k: v * self.box_predictor.loss_weight.get(k, 1.0) for k, v in losses.items()}

                # proposals is modified in-place below, so losses must be computed first.
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                        for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                return losses
            else:
                pred_instances, _ = self.box_predictor.inference(predictions, proposals)
                return pred_instances
    
    def sample_non_obj(self, energy_score_for_non_obj):
        # m = torch.distributions.Normal(0, 2)
        m = torch.distributions.Normal(energy_score_for_non_obj.min() + 1, 1)
        y = torch.exp(m.log_prob(energy_score_for_non_obj))
        weight = torch.nn.Softmax()(y).cpu().detach().numpy()
        index = np.random.choice(len(energy_score_for_non_obj), 100, p=weight)
        index = torch.from_numpy(index).cuda()
        weight = torch.from_numpy(weight).cuda()[index]
        return index, weight