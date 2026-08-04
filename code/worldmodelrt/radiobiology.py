from __future__ import annotations

from dataclasses import dataclass
from math import exp

import torch
from torch import nn

from worldmodelrt.schema import Schedule


@dataclass(frozen=True)
class Tissue:
    name: str
    alpha_beta: float
    tolerance_dose: float
    volume_exponent: float
    slope: float


TUMOR = Tissue("tumor", 10.0, 70.0, 0.12, 0.45)
BRAINSTEM = Tissue("brainstem", 2.0, 54.0, 0.16, 0.14)
PAROTID = Tissue("parotid", 3.0, 26.0, 0.70, 0.18)
OPTIC = Tissue("optic", 2.0, 55.0, 0.25, 0.14)
TEMPORAL = Tissue("temporal", 2.0, 60.0, 0.25, 0.15)


def survival_fraction(dose: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.exp(-alpha * dose - beta * dose.square())


def biologically_effective_dose(doses: torch.Tensor, alpha_beta: float) -> torch.Tensor:
    return (doses * (1.0 + doses / alpha_beta)).sum(dim=-1)


def equivalent_dose_2gy(doses: torch.Tensor, alpha_beta: float) -> torch.Tensor:
    return biologically_effective_dose(doses, alpha_beta) / (1.0 + 2.0 / alpha_beta)


def schedule_bed(schedule: Schedule, alpha_beta: float) -> float:
    schedule.validate()
    return sum(d * (1.0 + d / alpha_beta) for d in schedule.doses)


def oxygen_enhancement(oxygen: torch.Tensor, maximum: float = 3.0, half: float = 0.005) -> torch.Tensor:
    return (maximum * oxygen + half) / (oxygen + half)


def poisson_tcp(surviving_clonogens: torch.Tensor) -> torch.Tensor:
    return torch.exp(-surviving_clonogens.clamp_min(0.0))


def logistic_ntcp(equivalent_dose: torch.Tensor, tissue: Tissue) -> torch.Tensor:
    scale = max(tissue.slope * tissue.tolerance_dose, 1e-6)
    return torch.sigmoid((equivalent_dose - tissue.tolerance_dose) / scale)


def therapeutic_ratio(tcp: torch.Tensor, ntcp: torch.Tensor) -> torch.Tensor:
    return tcp / ntcp.clamp_min(1e-6)


def repopulation_penalty(elapsed_days: torch.Tensor, kick_off: float = 21.0, loss_per_day: float = 0.6) -> torch.Tensor:
    return (elapsed_days - kick_off).clamp_min(0.0) * loss_per_day


def gaussian_fraction_pulse(time: torch.Tensor, fraction_times: torch.Tensor, width: float) -> torch.Tensor:
    difference = time[..., None] - fraction_times
    return torch.exp(-0.5 * (difference / width).square())


class BoundedRadiobiology(nn.Module):
    def __init__(self, alpha: float, beta: float, ratio_min: float, ratio_max: float) -> None:
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(alpha).log())
        self.raw_beta = nn.Parameter(torch.tensor(beta).log())
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max

    @property
    def alpha(self) -> torch.Tensor:
        return self.raw_alpha.exp()

    @property
    def beta(self) -> torch.Tensor:
        alpha = self.alpha
        ratio = (alpha / self.raw_beta.exp()).clamp(self.ratio_min, self.ratio_max)
        return alpha / ratio

    @property
    def ratio(self) -> torch.Tensor:
        return self.alpha / self.beta

    def forward(self, dose: torch.Tensor, oxygen: torch.Tensor) -> torch.Tensor:
        effective_alpha = self.alpha * oxygen_enhancement(oxygen)
        return survival_fraction(dose, effective_alpha, self.beta)


def scalar_tcp(initial_cells: float, schedule: Schedule, alpha: float = 0.35, beta: float = 0.033) -> float:
    exponent = sum(alpha * dose + beta * dose * dose for dose in schedule.doses)
    surviving = initial_cells * exp(-exponent)
    return exp(-surviving)
