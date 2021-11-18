from torch import nn


class LabelSmoothedCrossEntropyCriterion(nn.Module):
    """Implement label smoothing."""

    def __init__(self, eps=0.1, reduce=True):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.eps = eps
        self.reduce = reduce

    def forward(self, lprobs, target):
        target = target.view(-1, 1)
        if self.reduce:
            nll_loss = -lprobs.gather(dim=-1, index=target)
            nll_loss = nll_loss.sum()
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            smooth_loss = smooth_loss.sum()
        else:
            nll_loss = -lprobs.gather(dim=-1, index=target)
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss
