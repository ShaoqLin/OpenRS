import torch
import torch.nn as nn
import torch.nn.functional as F


class ICLoss(nn.Module):
    """ Instance Contrastive Loss
    """
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
        self.prob_obj_head = ProbObjectnessHead(hidden_dim=128)

    def forward(self, features, labels, queue_features, queue_labels):
        device = features.device
        mask = torch.eq(labels[:, None], queue_labels[:, None].T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, queue_features.T), self.tau)

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
        
        # # objectness loss
        # objectness_loss = self.prob_obj_head(features)
        
        # loss
        # loss = - mean_log_prob_pos.mean() + objectness_loss
        loss = - mean_log_prob_pos.mean()
        # trick: avoid loss nan
        return loss if not torch.isnan(loss) else features.new_tensor(0.0)


class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0,1)
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        self.objectness_bn.eval()
        
    def forward(self, x):
        out=self.flatten(x)
        out=self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1)**2
    