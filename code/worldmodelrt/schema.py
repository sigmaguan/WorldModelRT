from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


@dataclass(frozen=True)
class FeatureLayout:
    tumor: int = 32
    spatial: int = 16
    clinical: int = 16
    action: int = 8

    @property
    def state(self) -> int:
        return self.tumor + self.spatial + self.clinical

    @property
    def observation_action(self) -> int:
        return self.state + self.action


@dataclass(frozen=True)
class ModelSpec:
    features: FeatureLayout = field(default_factory=FeatureLayout)
    latent_dim: int = 128
    transformer_layers: int = 4
    attention_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.1
    fourier_frequencies: int = 8
    alpha_initial: float = 0.35
    beta_initial: float = 0.033
    alpha_beta_min: float = 6.5
    alpha_beta_max: float = 29.0
    growth_initial: float = 0.05
    carrying_initial: float = 100.0
    reoxygenation_initial: float = 0.10
    consumption_initial: float = 0.05
    oxygen_maximum: float = 3.0
    oxygen_half: float = 0.005
    pulse_width: float = 0.03
    ode_tolerance: float = 1e-5


@dataclass(frozen=True)
class LossSpec:
    physics: float = 0.1
    kl: float = 0.01
    smooth: float = 0.001
    physics_margin: float = 0.1


@dataclass(frozen=True)
class StageSpec:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    clip_norm: float = 1.0
    warm_restart_period: int = 20


@dataclass(frozen=True)
class TrainingSpec:
    synthetic: StageSpec = field(default_factory=lambda: StageSpec("synthetic", 200, 256))
    population: StageSpec = field(default_factory=lambda: StageSpec("population", 100, 64))
    temporal: StageSpec = field(default_factory=lambda: StageSpec("temporal", 50, 16))
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    patience: int = 15
    seeds: tuple[int, ...] = (42, 123, 256, 389, 512, 678, 741, 853, 927, 1024)


@dataclass(frozen=True)
class Paths:
    data: Path
    runs: Path


@dataclass
class Batch:
    states: torch.Tensor
    actions: torch.Tensor
    times: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor
    fraction_times: torch.Tensor
    fraction_doses: torch.Tensor
    oxygen: torch.Tensor

    def to(self, device: torch.device) -> Batch:
        return Batch(*[value.to(device) for value in self.__dict__.values()])


@dataclass
class Prediction:
    observations: torch.Tensor
    latent: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_logvar: torch.Tensor
    derivatives: torch.Tensor
    lq_derivatives: torch.Tensor
    oxygen: torch.Tensor


@dataclass(frozen=True)
class Schedule:
    doses: tuple[float, ...]
    intervals: tuple[float, ...]
    label: str

    def validate(self) -> None:
        if not self.doses:
            raise ValueError("schedule has no fractions")
        if len(self.doses) != len(self.intervals):
            raise ValueError("dose and interval lengths differ")
        if min(self.doses) <= 0.0 or min(self.intervals) <= 0.0:
            raise ValueError("schedule values must be positive")

    @property
    def total_dose(self) -> float:
        return sum(self.doses)

    @property
    def elapsed_days(self) -> float:
        return sum(self.intervals)
