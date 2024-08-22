import logging
import os

import pytest

from prompto.scripts.run_experiment import (
    create_judge_experiment,
    load_env_file,
    load_judge_args,
    load_max_queries_json,
    parse_file_path_and_check_in_input,
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


def test_load_max_queries_json(temporary_data_folder_judge):
    # raise FileNotFoundError if file not found
    with pytest.raises(FileNotFoundError, match="File unknown.json not found"):
        loaded = load_max_queries_json("unknown.json")

    # raise ValueError if file is not json file path
    with pytest.raises(ValueError, match="max_queries_json must be a json file"):
        loaded = load_max_queries_json("test-exp-not-in-input.jsonl")

    loaded = load_max_queries_json("max_queries_dict.json")
    assert loaded == {"test": {"model1": 100, "model2": 120}}

    loaded = load_max_queries_json(max_queries_json=None)
    assert loaded == {}


def test_load_judge_args_both_none(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_judge_args(judge_location_arg=None, judge_arg=None)
    assert result == (False, None, None, None)
    assert (
        "Not creating judge file as one of judge_location or judge is None"
        in caplog.text
    )


def test_load_judge_args_judge_arg_none(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_judge_args(judge_location_arg="judge_loc", judge_arg=None)
    assert result == (False, None, None, None)
    assert (
        "Not creating judge file as one of judge_location or judge is None"
        in caplog.text
    )


def test_load_judge_args_judge_location_arg_none(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_judge_args(judge_location_arg=None, judge_arg="judge1")
    assert result == (False, None, None, None)
    assert (
        "Not creating judge file as one of judge_location or judge is None"
        in caplog.text
    )


def test_load_judge_args(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if both arguments are not None, return (True, templaate, judge_settings, judge
    result = load_judge_args(judge_location_arg="judge_loc", judge_arg="judge1,judge2")
    assert result == (
        True,
        "Template: input={INPUT_PROMPT}, output={OUTPUT_RESPONSE}",
        {
            "judge1": {
                "api": "test",
                "model_name": "model1",
                "parameters": {"temperature": 0.5},
            },
            "judge2": {
                "api": "test",
                "model_name": "model2",
                "parameters": {"temperature": 0.2, "top_k": 0.9},
            },
        },
        ["judge1", "judge2"],
    )
