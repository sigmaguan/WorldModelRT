from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from worldmodelrt.schema import Batch, FeatureLayout, Schedule


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    states: np.ndarray
    actions: np.ndarray
    times: np.ndarray
    targets: np.ndarray
    fraction_times: np.ndarray
    fraction_doses: np.ndarray
    oxygen: float


def standard_schedule() -> Schedule:
    intervals = tuple(3.0 if index % 5 == 4 else 1.0 for index in range(35))
    return Schedule((2.0,) * 35, intervals, "standard")


def accelerated_schedule() -> Schedule:
    intervals = tuple(2.0 if index % 6 == 5 else 1.0 for index in range(33))
    return Schedule((70.0 / 33.0,) * 33, intervals, "accelerated")


def hypofractionated_schedule() -> Schedule:
    intervals = tuple(3.0 if index % 5 == 4 else 1.0 for index in range(20))
    return Schedule((3.0,) * 20, intervals, "hypofractionated")


def interrupted_schedule() -> Schedule:
    intervals = list(standard_schedule().intervals)
    intervals[14] += 7.0
    return Schedule((2.0,) * 35, tuple(intervals), "interrupted")


def schedule_times(schedule: Schedule) -> np.ndarray:
    return np.cumsum(np.asarray((0.0,) + schedule.intervals[:-1], dtype=np.float32))


def lq_trajectory(initial_volume: float, schedule: Schedule, alpha: float, beta: float, doubling_days: float) -> np.ndarray:
    volumes = []
    surviving = initial_volume
    previous = 0.0
    for time, dose in zip(schedule_times(schedule), schedule.doses, strict=True):
        elapsed = float(time - previous)
        surviving *= math.exp(math.log(2.0) * elapsed / doubling_days)
        surviving *= math.exp(-alpha * dose - beta * dose * dose)
        volumes.append(surviving)
        previous = float(time)
    return np.asarray(volumes, dtype=np.float32)


class SyntheticCohort(Dataset[PatientRecord]):
    def __init__(self, size: int = 50000, seed: int = 42, features: FeatureLayout | None = None) -> None:
        self.size = size
        self.seed = seed
        self.features = features or FeatureLayout()
        self.schedules = (standard_schedule(), accelerated_schedule(), hypofractionated_schedule(), Schedule((2.0,) * 35, tuple(1.0 for _ in range(35)), "six_per_week"))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> PatientRecord:
        rng = np.random.default_rng(self.seed + index)
        alpha = rng.uniform(0.15, 0.55)
        beta = rng.uniform(0.01, 0.06)
        initial = rng.lognormal(math.log(30.0), 0.8)
        doubling = rng.uniform(3.0, 7.0)
        schedule = self.schedules[index % len(self.schedules)]
        times = schedule_times(schedule)
        volume = lq_trajectory(initial, schedule, alpha, beta, doubling)
        length = len(schedule.doses)
        states = rng.normal(0.0, 0.1, (length, self.features.state)).astype(np.float32)
        states[:, 0] = np.concatenate(([initial], volume[:-1]))
        states[:, 1] = alpha
        states[:, 2] = beta
        actions = np.zeros((length, self.features.action), dtype=np.float32)
        actions[:, 0] = schedule.doses
        actions[:, 1] = schedule.intervals
        targets = rng.normal(0.0, 0.03, (length, self.features.tumor)).astype(np.float32)
        targets[:, 0] = volume
        return PatientRecord(f"syn-{index:06d}", states, actions, times, targets, times, np.asarray(schedule.doses, dtype=np.float32), float(rng.uniform(0.05, 0.45)))


class JsonlCohort(Dataset[PatientRecord]):
    def __init__(self, manifest: Path) -> None:
        self.root = manifest.parent
        with manifest.open(encoding="utf-8") as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> PatientRecord:
        row = self.rows[index]
        archive = np.load(self.root / row["archive"], allow_pickle=False)
        return PatientRecord(
            str(row["patient_id"]),
            archive["states"],
            archive["actions"],
            archive["times"],
            archive["targets"],
            archive["fraction_times"],
            archive["fraction_doses"],
            float(archive["oxygen"]),
        )


def collate(records: Sequence[PatientRecord]) -> Batch:
    maximum = max(len(record.times) for record in records)
    batch = len(records)
    state_dim = records[0].states.shape[-1]
    action_dim = records[0].actions.shape[-1]
    target_dim = records[0].targets.shape[-1]
    fraction_maximum = max(len(record.fraction_times) for record in records)
    states = torch.zeros(batch, maximum, state_dim)
    actions = torch.zeros(batch, maximum, action_dim)
    times = torch.zeros(batch, maximum)
    targets = torch.zeros(batch, maximum, target_dim)
    mask = torch.zeros(batch, maximum, dtype=torch.bool)
    fraction_times = torch.zeros(batch, fraction_maximum)
    fraction_doses = torch.zeros(batch, fraction_maximum)
    oxygen = torch.zeros(batch, 1)
    for index, record in enumerate(records):
        length = len(record.times)
        fractions = len(record.fraction_times)
        states[index, :length] = torch.from_numpy(record.states)
        actions[index, :length] = torch.from_numpy(record.actions)
        times[index, :length] = torch.from_numpy(record.times)
        targets[index, :length] = torch.from_numpy(record.targets)
        mask[index, :length] = True
        fraction_times[index, :fractions] = torch.from_numpy(record.fraction_times)
        fraction_doses[index, :fractions] = torch.from_numpy(record.fraction_doses)
        oxygen[index, 0] = record.oxygen
    return Batch(states, actions, times, targets, mask, fraction_times, fraction_doses, oxygen)


def patient_split(identifiers: Sequence[str], seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    order = np.arange(len(identifiers))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    train_end = int(0.8 * len(order))
    validation_end = int(0.9 * len(order))
    return order[:train_end].tolist(), order[train_end:validation_end].tolist(), order[validation_end:].tolist()


def read_clinical_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            yield {str(key).strip(): str(value).strip() for key, value in row.items()}


def normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean) / np.maximum(scale, 1e-6)


def statistics(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate(arrays, axis=0)
    return joined.mean(axis=0), joined.std(axis=0)
