# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    # Main absolute loss in centimeters
    w_huber: float = 1.0
    huber_delta_cm: float = 20.0

    # Relative-error regularizer
    w_absrel: float = 0.25

    # Small structure stabilizer only
    w_silog: float = 0.10
    silog_lambda: float = 0.15

    # Optional edge regularizer
    w_grad: float = 0.0

    eps: float = 1e-6


class SiLogLoss(nn.Module):
    def __init__(self, lambd: float = 0.15, eps: float = 1e-6):
        super().__init__()
        self.lambd = float(lambd)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        vm = valid_mask.bool()
        if int(vm.sum().item()) == 0:
            return pred.new_tensor(0.0)

        pred_v = pred[vm].clamp_min(self.eps)
        tgt_v = target[vm].clamp_min(self.eps)

        d = torch.log(tgt_v) - torch.log(pred_v)
        mean_sq = (d * d).mean()
        mean = d.mean()
        val = mean_sq - self.lambd * (mean * mean)
        return torch.sqrt(torch.clamp(val, min=0.0))


class GradientLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        pred/target: [B, H, W]
        valid_mask:  [B, H, W]
        """
        if pred.ndim != 3:
            raise ValueError(f"Expected [B,H,W], got {tuple(pred.shape)}")

        dx_p = pred[:, :, 1:] - pred[:, :, :-1]
        dx_t = target[:, :, 1:] - target[:, :, :-1]
        dx_v = valid_mask[:, :, 1:] & valid_mask[:, :, :-1]

        dy_p = pred[:, 1:, :] - pred[:, :-1, :]
        dy_t = target[:, 1:, :] - target[:, :-1, :]
        dy_v = valid_mask[:, 1:, :] & valid_mask[:, :-1, :]

        loss_x = self._masked_l1(dx_p, dx_t, dx_v)
        loss_y = self._masked_l1(dy_p, dy_t, dy_v)
        return 0.5 * (loss_x + loss_y)

    def _masked_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if int(valid_mask.sum().item()) == 0:
            return pred.new_tensor(0.0)
        diff = torch.abs(pred[valid_mask] - target[valid_mask])
        return diff.mean()


class AbsoluteMetricLoss(nn.Module):
    """
    Absolute-depth loss for metric fine-tuning.

    Main term: Huber in cm
    Auxiliary terms: AbsRel + small SiLog (+ optional gradient)
    """

    def __init__(self, cfg: Optional[LossConfig] = None):
        super().__init__()
        self.cfg = cfg or LossConfig()
        self.silog = SiLogLoss(lambd=self.cfg.silog_lambda, eps=self.cfg.eps)
        self.grad_loss = GradientLoss(eps=self.cfg.eps)

    def forward(
        self,
        pred_cm: torch.Tensor,
        gt_cm: torch.Tensor,
        valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        v = valid & torch.isfinite(pred_cm) & torch.isfinite(gt_cm) & (gt_cm > 0)
        if int(v.sum().item()) == 0:
            z = pred_cm.sum() * 0.0
            return {
                "loss": z,
                "huber": z.detach(),
                "absrel": z.detach(),
                "silog": z.detach(),
                "grad": z.detach(),
            }

        pred_v = pred_cm[v].clamp_min(cfg.eps)
        gt_v = gt_cm[v].clamp_min(cfg.eps)

        huber = F.smooth_l1_loss(
            pred_v,
            gt_v,
            beta=cfg.huber_delta_cm,
            reduction="mean",
        )
        absrel = (torch.abs(pred_v - gt_v) / gt_v).mean()
        silog = self.silog(pred_cm, gt_cm, v)
        grad = self.grad_loss(pred_cm, gt_cm, v) if cfg.w_grad > 0 else pred_cm.new_tensor(0.0)

        loss = (
            cfg.w_huber * huber +
            cfg.w_absrel * absrel +
            cfg.w_silog * silog +
            cfg.w_grad * grad
        )

        return {
            "loss": loss,
            "huber": huber.detach(),
            "absrel": absrel.detach(),
            "silog": silog.detach(),
            "grad": grad.detach(),
        }