from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def relative_volume_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).abs().div(target.abs().clamp_min(1e-6)).mean() * 100.0


def dice(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    left = prediction >= threshold
    right = target >= threshold
    intersection = (left & right).flatten(1).sum(1)
    denominator = left.flatten(1).sum(1) + right.flatten(1).sum(1)
    return ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def pearson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left = prediction - prediction.mean(dim=-1, keepdim=True)
    right = target - target.mean(dim=-1, keepdim=True)
    numerator = (left * right).sum(dim=-1)
    denominator = left.square().sum(dim=-1).sqrt() * right.square().sum(dim=-1).sqrt()
    return (numerator / denominator.clamp_min(1e-8)).mean()


def concordance_index(times: np.ndarray, scores: np.ndarray, events: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    for left in range(len(times)):
        for right in range(left + 1, len(times)):
            if times[left] == times[right]:
                continue
            earlier, later = (left, right) if times[left] < times[right] else (right, left)
            if events[earlier] == 0:
                continue
            comparable += 1.0
            if scores[earlier] > scores[later]:
                concordant += 1.0
            elif scores[earlier] == scores[later]:
                concordant += 0.5
    return concordant / comparable if comparable else float("nan")


def treatment_sensitivity(standard: torch.Tensor, alternatives: torch.Tensor) -> torch.Tensor:
    distances = (alternatives - standard[:, None]).square().mean(dim=(-1, -2)).sqrt()
    return distances.mean()


def violation_rate(volumes: torch.Tensor, doses: torch.Tensor, tolerance: float = 0.02) -> torch.Tensor:
    change = volumes[:, 1:] - volumes[:, :-1]
    delivered = doses[:, 1:] > 0.0
    return ((change > tolerance) & delivered).float().sum() / delivered.float().sum().clamp_min(1.0)


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float


def bootstrap(values: np.ndarray, resamples: int = 1000, seed: int = 42) -> BootstrapInterval:
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        samples[index] = rng.choice(values, size=len(values), replace=True).mean()
    return BootstrapInterval(float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    difference = left - right
    return float(difference.mean() / difference.std(ddof=1))


def holm_bonferroni(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, float(values[index]) * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)[::-1]
    adjusted = np.empty_like(values)
    running = 1.0
    count = len(values)
    for reverse_rank, index in enumerate(order):
        rank = count - reverse_rank
        running = min(running, float(values[index]) * count / rank)
        adjusted[index] = running
    return adjusted
