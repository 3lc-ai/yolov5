# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions per sample - WIP
"""
import torch
import torch.nn as nn

from ..loss import ComputeLoss, FocalLoss, smooth_BCE


class TLCComputeLoss(ComputeLoss):
    # Compute losses
    def __init__(self, device, h, stride, na, nc, nl, anchors, autobalance=False):

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = na  # number of anchors
        self.nc = nc  # number of classes
        self.nl = nl  # number of layers
        self.device = device

        self.anchors = torch.clone(anchors).detach().to(self.device)
