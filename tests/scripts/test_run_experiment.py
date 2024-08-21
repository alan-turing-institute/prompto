import logging
import os

import pytest

from prompto.scripts.run_experiment import (
    create_judge_experiment,
    load_env_file,
    load_judge_args,
    load_max_queries_json,
    parse_file_path,
)


def test_load_env_file(temporary_data_folders, caplog):
    caplog.set_level(logging.INFO)

    assert "TEST_ENV_VAR" not in os.environ
    loaded = load_env_file(".env")
    assert loaded
    assert os.environ["TEST_ENV_VAR"] == "test"

    assert "Loaded environment variables from .env" in caplog.text


def test_load_env_file_not_found(caplog):
    caplog.set_level(logging.INFO)

    loaded = load_env_file(".env")
    assert not loaded
    assert "No environment file found at .env" in caplog.text
