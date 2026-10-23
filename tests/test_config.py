"""Your first test: project_config.yml loads with the course's ProjectConfig for dev.

`ci.yml` runs every test under tests/ on each pull request and each push to main; add yours here.
"""

from pathlib import Path

import pytest

from arxiv_curator.config import ProjectConfig, get_env

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


def test_project_config_loads_for_dev() -> None:
    """Load the provisioned settings without requiring optional lecture resources."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), "dev")

    assert cfg.full_volume_path == "llmops_dev.mahsa_keshtkarsohi.arxiv_files"
    assert cfg.warehouse_id == "be8f0e8cc38ba3f4"


def test_get_env_defaults_to_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a widget or ENV variable the environment is dev."""
    monkeypatch.delenv("ENV", raising=False)

    assert get_env() == "dev"


def test_get_env_reads_env_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ENV variable selects the environment when there is no widget."""
    monkeypatch.setenv("ENV", "acc")

    assert get_env() == "acc"


def test_get_env_falls_back_when_widget_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Spark session that cannot provide the env widget falls back to ENV."""
    monkeypatch.setenv("ENV", "prd")

    assert get_env(object()) == "prd"
