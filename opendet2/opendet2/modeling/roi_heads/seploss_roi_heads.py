# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
    add_ground_truth_to_proposals,
)
from detectron2.structures import Instances, pairwise_iou, Boxes, ImageList
from detectron2.utils.events import get_event_storage

from .fast_rcnn import build_roi_box_output_layers

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class SeplossOpenSetStandardROIHeads(StandardROIHeads):
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
            # if match_quality_matrix.shape[0]:
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
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
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

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
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
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
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
            np.random.shuffle(y)    # why not shuffle x ?
            proposals_idx = propid_matched_thisgt[0][x]
            proposals_idy = propid_matched_thisgt[0][y]
            proposals_x = pos_proposal_boxes[proposals_idx]
            proposals_y = pos_proposal_boxes[proposals_idy]
            iou_x = pos_iou_with_gt[proposals_idx]
            iou_y = pos_iou_with_gt[proposals_idy]
            # sort
            sort_x = iou_x.sort()[1]    # sort_x, indices = torch.sort(iou_x) 
            sort_y = iou_y.sort()[1]    # here we want indices.
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

    def sample_localboxes_and_completeboxes_t2(self, proposals, targets):
        proposals_batch_pos_x = []
        proposals_batch_pos_y = []
        bool_pos_batch = []
        # batch_instances_ids = [] # gt'id that proposals belong to 
        e1 = 0.0 # 0.0
        e2 = 0.5 # 0.5
        for i, img in enumerate(proposals):
            proposal_boxes = img.proposal_boxes
            gt_boxes = targets[i].gt_boxes
            gt_num = targets[i].gt_classes.shape[0]
            iou_matrix = pairwise_iou(gt_boxes, proposal_boxes) # gt * pro
            
            iou_with_gt, match_gtid = iou_matrix.max(0)
            potential_index = torch.where(iou_with_gt > e2) 
            proposal_boxes = proposal_boxes[potential_index] # 更新proposals, iou, match_gt_id
            iou_with_gt = iou_with_gt[potential_index]
            match_gtid = match_gtid[potential_index]

            pos_x_proposals, pos_y_proposals, pos_bool_list = self.export_pos_box(proposal_boxes, iou_with_gt, match_gtid, gt_num, e2)
            
            
            proposals_batch_pos_x.append(Boxes.cat(pos_x_proposals)) 
            proposals_batch_pos_y.append(Boxes.cat(pos_y_proposals))    
            bool_pos_batch.append(torch.cat(pos_bool_list))
            

        return proposals_batch_pos_x, proposals_batch_pos_y, bool_pos_batch
