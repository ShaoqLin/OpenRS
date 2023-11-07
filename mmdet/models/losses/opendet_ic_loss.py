import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from torch import Tensor
from mmdet.registry import MODELS

@MODELS.register_module()
class ICLoss(nn.Module):

    def __init__(self,
                 ic_loss_out_dim,
                 ic_loss_queue_size,
                 ic_loss_in_queue_size,
                 ic_loss_batch_iou_thr,
                 ic_loss_queue_iou_thr,
                 ic_loss_queue_tau,
                 ic_loss_weight):

        super(ICLoss, self).__init__()
        self.ic_loss_out_dim = ic_loss_out_dim
        self.ic_loss_queue_size = ic_loss_queue_size
        self.ic_loss_in_queue_size = ic_loss_in_queue_size
        self.ic_loss_batch_iou_thr = ic_loss_batch_iou_thr
        self.ic_loss_queue_iou_thr = ic_loss_queue_iou_thr
        self.ic_loss_queue_tau = ic_loss_queue_tau
        self.ic_loss_weight = ic_loss_weight

    def forward(self, features, labels, queue_features, queue_labels):
        device = features.device
        mask = torch.eq(labels[:, None], queue_labels[:, None].T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, queue_features.T), self.ic_loss_queue_tau)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(logits)
        # mask itself
        logits_mask[logits == 0] = 0

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - mean_log_prob_pos.mean()
        # trick: avoid loss nan
        return loss if not torch.isnan(loss) else features.new_tensor(0.0)


    # def forward(self, scores: Tensor, labels: Tensor):
    #     fg_scores, bg_scores, fg_labels, bg_labels = self._sampling(
    #         scores, labels)
    #     # sample both fg and bg
    #     scores = torch.cat([fg_scores, bg_scores])
    #     labels = torch.cat([fg_labels, bg_labels])

    #     num_sample, num_classes = scores.shape
    #     mask = torch.arange(num_classes).repeat(
    #         num_sample, 1).to(scores.device)
    #     inds = mask != labels[:, None].repeat(1, num_classes)
    #     mask = mask[inds].reshape(num_sample, num_classes-1)

    #     gt_scores = torch.gather(
    #         F.softmax(scores, dim=1), 1, labels[:, None]).squeeze(1)
    #     mask_scores = torch.gather(scores, 1, mask)

    #     gt_scores[gt_scores < 0] = 0.0
    #     targets = torch.zeros_like(mask_scores)
    #     num_fg = fg_scores.size(0)
    #     targets[:num_fg, self.num_classes-2] = gt_scores[:num_fg] * \
    #         (1-gt_scores[:num_fg]).pow(self.alpha)
    #     targets[num_fg:, self.num_classes-1] = gt_scores[num_fg:] * \
    #         (1-gt_scores[num_fg:]).pow(self.alpha)

    #     return self._soft_cross_entropy(mask_scores, targets.detach())
