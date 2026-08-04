from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from worldmodelrt.model import WorldModelRT
from worldmodelrt.objectives import WorldModelLoss
from worldmodelrt.schema import Batch, StageSpec

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_loss: float = float("inf")
    stale_epochs: int = 0
    seed: int = 42


class AtomicStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, payload: object) -> Path:
        target = self.directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        return target

    def load(self, name: str, device: torch.device) -> dict[str, object]:
        value = torch.load(self.directory / name, map_location=device, weights_only=False)
        if not isinstance(value, dict):
            raise TypeError("stored training state is not a mapping")
        return value


class Trainer:
    def __init__(self, model: WorldModelRT, stage: StageSpec, device: torch.device, output: Path, seed: int = 42) -> None:
        self.model = model.to(device)
        self.stage = stage
        self.device = device
        self.loss = WorldModelLoss().to(device)
        self.optimizer = AdamW(model.parameters(), lr=stage.learning_rate, weight_decay=stage.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, stage.warm_restart_period)
        self.store = AtomicStore(output)
        self.state = TrainState(seed=seed)
        set_seed(seed)

    def train_batch(self, batch: Batch) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(batch.to(self.device), stochastic=True)
        report = self.loss(prediction, batch.to(self.device))
        report.total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.stage.clip_norm)
        self.optimizer.step()
        self.state.step += 1
        return report.detached()

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Batch]) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        for batch in loader:
            moved = batch.to(self.device)
            total += float(self.loss(self.model(moved, stochastic=False), moved).total)
            count += 1
        return total / max(count, 1)

    def checkpoint(self, name: str = "latest.pt") -> Path:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": asdict(self.state),
            "stage": asdict(self.stage),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        }
        return self.store.save(name, payload)

    def restore(self, name: str = "latest.pt") -> None:
        payload = self.store.load(name, self.device)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.state = TrainState(**payload["state"])
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])

    def fit(self, train: Iterable[Batch], validation: Iterable[Batch], patience: int = 15) -> TrainState:
        for epoch in range(self.state.epoch, self.stage.epochs):
            self.state.epoch = epoch
            for batch in train:
                report = self.train_batch(batch)
                if self.state.step % 100 == 0:
                    LOGGER.info("stage=%s epoch=%d step=%d loss=%.6f", self.stage.name, epoch, self.state.step, report["total"])
            validation_loss = self.evaluate(validation)
            self.scheduler.step(epoch + 1)
            if validation_loss < self.state.best_loss:
                self.state.best_loss = validation_loss
                self.state.stale_epochs = 0
                self.checkpoint("best.pt")
            else:
                self.state.stale_epochs += 1
            self.checkpoint()
            if self.state.stale_epochs >= patience:
                break
        return self.state


def write_run_record(path: Path, state: TrainState, stage: StageSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"state": asdict(state), "stage": asdict(stage)}, indent=2), encoding="utf-8")
    os.replace(temporary, path)
