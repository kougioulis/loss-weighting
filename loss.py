import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_device


class VanillaMultiLoss(nn.Module):
    def __init__(self, n_losses: int): 
        super().__init__()
        # Learnable scaling factors for each loss term
        # Initialized at zero so exp / softplus starts near neutral weighting
        self.loss_term_scaling = nn.Parameter(
            torch.zeros(n_losses, device=get_device())
        )

    def forward(self, losses: list) -> torch.Tensor:
        # Simple weighted sum of losses with positive scalings
        total_loss = torch.zeros(
            1, device=losses[0].device, dtype=losses[0].dtype
        )

        for i, loss in enumerate(losses):
            scaling = F.softplus(self.loss_term_scaling[i])  # ensure > 0
            total_loss += scaling * loss

        return total_loss


class MultiNoiseLoss(nn.Module):
    """
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    (Kendall et al., CVPR 2018).
    """
    def __init__(self, n_losses: int):
        super(MultiNoiseLoss, self).__init__()
        # Learnable log-variances (log sigma^2), which is the canonical and numerically
        # stable parametrization used in the paper and most reference implementations
        self.log_vars = nn.Parameter(
            torch.zeros(n_losses, device=get_device())
        )

    def forward(self, losses: list) -> torch.Tensor:
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)

        Each loss coeff is of the form:
            1 / sigma_i^2 * ell_i + log(sigma_i^2)

        Using the log-variance parametrization:
            precision_i = exp(-log_var_i)

        Total loss:
            ell = sum_i [ exp(-log_var_i) * ell_i + log_var_i ]
        """
        total_loss = torch.zeros(
            1, device=losses[0].device, dtype=losses[0].dtype
        )

        for i, loss in enumerate(losses):
            precision_i = torch.exp(-self.log_vars[i])
            total_loss += precision_i * loss + self.log_vars[i]

        return total_loss
