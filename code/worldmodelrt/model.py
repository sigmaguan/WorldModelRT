from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from worldmodelrt.radiobiology import BoundedRadiobiology, gaussian_fraction_pulse, oxygen_enhancement
from worldmodelrt.schema import Batch, ModelSpec, Prediction


class ContinuousEncoding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        powers = torch.arange(0, dimension, 2, dtype=torch.float32) / dimension
        self.register_buffer("scales", torch.pow(10000.0, powers), persistent=False)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        phases = times[..., None] / self.scales
        result = torch.zeros(*times.shape, self.dimension, device=times.device, dtype=times.dtype)
        result[..., 0::2] = phases.sin()
        result[..., 1::2] = phases.cos()
        return result


class ObservationProjector(nn.Module):
    def __init__(self, source: int, target: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(source, target * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(target * 2, target))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class TimeAwareEncoder(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.project = ObservationProjector(spec.features.observation_action, spec.latent_dim, spec.dropout)
        self.position = ContinuousEncoding(spec.latent_dim)
        layer = nn.TransformerEncoderLayer(spec.latent_dim, spec.attention_heads, spec.feedforward_dim, spec.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, spec.transformer_layers, nn.LayerNorm(spec.latent_dim))
        self.posterior = nn.Sequential(nn.Linear(spec.latent_dim, spec.latent_dim * 2), nn.GELU(), nn.Linear(spec.latent_dim * 2, spec.latent_dim * 2))

    def forward(self, states: torch.Tensor, actions: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.project(torch.cat((states, actions), dim=-1)) + self.position(times)
        hidden = self.transformer(embedded, src_key_padding_mask=~mask.bool())
        index = mask.long().sum(dim=1).sub(1).clamp_min(0)
        selected = hidden[torch.arange(hidden.shape[0], device=hidden.device), index]
        return self.posterior(selected).chunk(2, dim=-1)

    @staticmethod
    def sample(mean: torch.Tensor, logvar: torch.Tensor, stochastic: bool) -> torch.Tensor:
        if not stochastic:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar.clamp(-20.0, 8.0))


class FourierClock(nn.Module):
    def __init__(self, frequencies: int) -> None:
        super().__init__()
        values = torch.logspace(math.log10(1.0 / 35.0), math.log10(1.0), frequencies)
        self.register_buffer("frequencies", values, persistent=True)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * torch.pi * time[..., None] * self.frequencies
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class OxygenDynamics(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.raw_reoxygenation = nn.Parameter(torch.tensor(spec.reoxygenation_initial).log())
        self.raw_consumption = nn.Parameter(torch.tensor(spec.consumption_initial).log())

    def forward(self, oxygen: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        reoxygenation = self.raw_reoxygenation.exp()
        consumption = self.raw_consumption.exp()
        burden = latent.abs().mean(dim=-1, keepdim=True)
        return reoxygenation * (1.0 - oxygen) - consumption * oxygen * burden


class LatentDynamics(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.clock = FourierClock(spec.fourier_frequencies)
        inputs = spec.latent_dim + spec.features.action + 2 * spec.fourier_frequencies + 1
        self.residual = nn.Sequential(
            nn.Linear(inputs, 2 * spec.latent_dim), nn.GELU(), nn.Linear(2 * spec.latent_dim, spec.latent_dim), nn.GELU(), nn.Linear(spec.latent_dim, spec.latent_dim)
        )
        self.lq_projection = nn.Linear(spec.latent_dim, spec.latent_dim, bias=False)
        self.radiobiology = BoundedRadiobiology(spec.alpha_initial, spec.beta_initial, spec.alpha_beta_min, spec.alpha_beta_max)
        self.raw_growth = nn.Parameter(torch.tensor(spec.growth_initial).log())
        self.raw_carrying = nn.Parameter(torch.tensor(spec.carrying_initial).log())
        self.oxygen = OxygenDynamics(spec)

    def components(
        self, time: torch.Tensor, latent: torch.Tensor, action: torch.Tensor, oxygen: torch.Tensor, fraction_times: torch.Tensor, fraction_doses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pulse = gaussian_fraction_pulse(time.squeeze(-1), fraction_times, self.spec.pulse_width)
        weighted_dose = (pulse * fraction_doses).sum(dim=-1, keepdim=True)
        alpha = self.radiobiology.alpha * oxygen_enhancement(oxygen, self.spec.oxygen_maximum, self.spec.oxygen_half)
        effect = alpha * weighted_dose + self.radiobiology.beta * weighted_dose.square()
        lq = -self.lq_projection(latent) * effect
        clock = self.clock(time.squeeze(-1))
        neural = self.residual(torch.cat((latent, action, clock, oxygen), dim=-1))
        norm = latent.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
        growth = self.raw_growth.exp() * latent * torch.log(self.raw_carrying.exp() / norm + 1e-6)
        oxygen_rate = self.oxygen(oxygen, latent)
        return lq, neural, growth, oxygen_rate

    def forward(
        self, time: torch.Tensor, latent: torch.Tensor, action: torch.Tensor, oxygen: torch.Tensor, fraction_times: torch.Tensor, fraction_doses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lq, neural, growth, oxygen_rate = self.components(time, latent, action, oxygen, fraction_times, fraction_doses)
        return lq + neural + growth, oxygen_rate, lq


class EventIntegrator(nn.Module):
    def __init__(self, dynamics: LatentDynamics, tolerance: float) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.tolerance = tolerance

    def step(
        self, time: torch.Tensor, dt: torch.Tensor, latent: torch.Tensor, action: torch.Tensor, oxygen: torch.Tensor, fraction_times: torch.Tensor, fraction_doses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k1, o1, lq = self.dynamics(time, latent, action, oxygen, fraction_times, fraction_doses)
        k2, o2, _ = self.dynamics(time + dt / 2, latent + dt * k1 / 2, action, oxygen + dt * o1 / 2, fraction_times, fraction_doses)
        k3, o3, _ = self.dynamics(time + dt / 2, latent + dt * k2 / 2, action, oxygen + dt * o2 / 2, fraction_times, fraction_doses)
        k4, o4, _ = self.dynamics(time + dt, latent + dt * k3, action, oxygen + dt * o3, fraction_times, fraction_doses)
        next_latent = latent + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        next_oxygen = (oxygen + dt * (o1 + 2 * o2 + 2 * o3 + o4) / 6).clamp(0.0, 1.0)
        return next_latent, next_oxygen, k1, lq

    def forward(
        self, initial: torch.Tensor, oxygen: torch.Tensor, times: torch.Tensor, actions: torch.Tensor, fraction_times: torch.Tensor, fraction_doses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = initial
        current_oxygen = oxygen
        states = []
        oxygens = []
        derivatives = []
        lq_derivatives = []
        for index in range(times.shape[1]):
            dt = torch.zeros_like(times[:, index : index + 1]) if index == 0 else (times[:, index : index + 1] - times[:, index - 1 : index]).clamp_min(1e-4)
            latent, current_oxygen, derivative, lq = self.step(times[:, index : index + 1], dt, latent, actions[:, index], current_oxygen, fraction_times, fraction_doses)
            states.append(latent)
            oxygens.append(current_oxygen)
            derivatives.append(derivative)
            lq_derivatives.append(lq)
        return torch.stack(states, 1), torch.stack(oxygens, 1), torch.stack(derivatives, 1), torch.stack(lq_derivatives, 1)


class ObservationDecoder(nn.Module):
    def __init__(self, latent: int, output: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(latent, latent * 2), nn.GELU(), nn.Linear(latent * 2, output))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        output = self.layers(latent)
        volume = F.softplus(output[..., :1])
        return torch.cat((volume, output[..., 1:]), dim=-1)


class WorldModelRT(nn.Module):
    def __init__(self, spec: ModelSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or ModelSpec()
        self.encoder = TimeAwareEncoder(self.spec)
        self.dynamics = LatentDynamics(self.spec)
        self.integrator = EventIntegrator(self.dynamics, self.spec.ode_tolerance)
        self.decoder = ObservationDecoder(self.spec.latent_dim, self.spec.features.tumor)

    def forward(self, batch: Batch, stochastic: bool = True) -> Prediction:
        mean, logvar = self.encoder(batch.states, batch.actions, batch.times, batch.mask)
        initial = self.encoder.sample(mean, logvar, stochastic)
        latent, oxygen, derivatives, lq = self.integrator(initial, batch.oxygen, batch.times, batch.actions, batch.fraction_times, batch.fraction_doses)
        return Prediction(self.decoder(latent), latent, mean, logvar, derivatives, lq, oxygen)
