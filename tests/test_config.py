"""Tests for package import and configuration loading."""

from pathlib import Path

import pytest

import churn
from churn.config import Config, load_config


def test_package_imports():
    """The package imports and exposes a version string."""
    assert isinstance(churn.__version__, str)
    assert churn.__version__


def test_load_default_config():
    """The default config.yaml loads into a typed Config object."""
    config = load_config()
    assert isinstance(config, Config)

    # Spot-check values from each section.
    assert config.project.random_seed == 42
    assert config.data.target_column == "Churn"
    assert config.data.id_column == "customerID"
    assert config.data.positive_label == "Yes"
    assert 0.0 < config.split.test_size < 1.0
    assert config.model.selection_metric == "roc_auc"
    assert config.reports.figures_dir
    assert config.api.port == 8000


def test_report_paths_are_absolute():
    """Figure and metrics paths resolve to absolute locations."""
    config = load_config()
    assert config.figures_path.is_absolute()
    assert config.metrics_path.is_absolute()
    assert config.metrics_path.name.endswith(".json")


def test_model_path_is_absolute():
    """The convenience model_path property resolves to an absolute path."""
    config = load_config()
    assert config.model_path.is_absolute()
    assert config.model_path.name.endswith(".joblib")


def test_missing_config_raises(tmp_path: Path):
    """A missing config file raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)
