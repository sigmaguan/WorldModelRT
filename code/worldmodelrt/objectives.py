from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from worldmodelrt.schema import Batch, LossSpec, Prediction


@dataclass
class LossReport:
    total: torch.Tensor
    reconstruction: torch.Tensor
    physics: torch.Tensor
    kl: torch.Tensor
    smoothness: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.__dict__.items()}


class WorldModelLoss(nn.Module):
    def __init__(self, spec: LossSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or LossSpec()

    def reconstruction(self, prediction: Prediction, batch: Batch) -> torch.Tensor:
        weights = batch.mask[..., None].to(prediction.observations.dtype)
        error = (prediction.observations - batch.targets).square() * weights
        return error.sum() / weights.sum().clamp_min(1.0)

    def physics(self, prediction: Prediction) -> torch.Tensor:
        departure = (prediction.derivatives - prediction.lq_derivatives).norm(dim=-1)
        return torch.relu(departure - self.spec.physics_margin).square().mean()

    def kl(self, prediction: Prediction) -> torch.Tensor:
        value = -0.5 * (1.0 + prediction.posterior_logvar - prediction.posterior_mean.square() - prediction.posterior_logvar.exp())
        return value.sum(dim=-1).mean()

    def smoothness(self, prediction: Prediction) -> torch.Tensor:
        if prediction.latent.shape[1] < 3:
            return prediction.latent.new_zeros(())
        second = prediction.latent[:, 2:] - 2.0 * prediction.latent[:, 1:-1] + prediction.latent[:, :-2]
        return second.square().mean()

    def forward(self, prediction: Prediction, batch: Batch) -> LossReport:
        reconstruction = self.reconstruction(prediction, batch)
        physics = self.physics(prediction)
        kl = self.kl(prediction)
        smoothness = self.smoothness(prediction)
        total = reconstruction + self.spec.physics * physics + self.spec.kl * kl + self.spec.smooth * smoothness
        return LossReport(total, reconstruction, physics, kl, smoothness)
