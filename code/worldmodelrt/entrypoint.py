from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from worldmodelrt.data import SyntheticCohort, collate, patient_split
from worldmodelrt.model import WorldModelRT
from worldmodelrt.schema import StageSpec
from worldmodelrt.training import Trainer


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="worldmodelrt-train")
    value.add_argument("--stage", choices=("synthetic", "population", "temporal"), required=True)
    value.add_argument("--data", type=Path, default=Path("data"))
    value.add_argument("--output", type=Path, default=Path("runs"))
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--workers", type=int, default=8)
    return value


def stage_spec(name: str) -> StageSpec:
    if name == "synthetic":
        return StageSpec(name, 200, 256)
    if name == "population":
        return StageSpec(name, 100, 64)
    return StageSpec(name, 50, 16)


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arguments.stage != "synthetic":
        raise RuntimeError("clinical stages require prepared institution-specific manifests")
    cohort = SyntheticCohort(seed=arguments.seed)
    identifiers = [f"syn-{index:06d}" for index in range(len(cohort))]
    train_indices, validation_indices, _ = patient_split(identifiers, arguments.seed)
    specification = stage_spec(arguments.stage)
    train_loader = DataLoader(Subset(cohort, train_indices), batch_size=specification.batch_size, shuffle=True, num_workers=arguments.workers, collate_fn=collate, pin_memory=True)
    validation_loader = DataLoader(Subset(cohort, validation_indices), batch_size=specification.batch_size, num_workers=arguments.workers, collate_fn=collate, pin_memory=True)
    trainer = Trainer(WorldModelRT(), specification, device, arguments.output / arguments.stage, arguments.seed)
    trainer.fit(train_loader, validation_loader)


if __name__ == "__main__":
    main()
