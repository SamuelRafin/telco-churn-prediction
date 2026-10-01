"""Loads config.yaml into dataclasses so the rest of the code gets attribute
access and a typo raises instead of returning None.

Frozen because nothing should be rewriting the seed partway through a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    processed_path: str
    target_column: str
    id_column: str
    positive_label: str


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    stratify: bool


@dataclass(frozen=True)
class ModelConfig:
    artifacts_dir: str
    model_file: str
    selection_metric: str
    decision_threshold: float


@dataclass(frozen=True)
class ReportsConfig:
    figures_dir: str
    metrics_file: str


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    reports: ReportsConfig
    api: ApiConfig

    @property
    def model_path(self) -> Path:
        return PROJECT_ROOT / self.model.artifacts_dir / self.model.model_file

    @property
    def figures_path(self) -> Path:
        return PROJECT_ROOT / self.reports.figures_dir

    @property
    def metrics_path(self) -> Path:
        return PROJECT_ROOT / self.reports.metrics_file


def load_config(path: str | Path | None = None) -> Config:
    """Read the YAML config. Defaults to config.yaml at the project root."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return Config(
        project=ProjectConfig(**raw["project"]),
        data=DataConfig(**raw["data"]),
        split=SplitConfig(**raw["split"]),
        model=ModelConfig(**raw["model"]),
        reports=ReportsConfig(**raw["reports"]),
        api=ApiConfig(**raw["api"]),
    )
