import logging
import os

import pytest

from prompto.experiment import Experiment
from prompto.scripts.run_experiment import (
    create_judge_experiment,
    load_env_file,
    load_judge_args,
    load_max_queries_json,
    parse_file_path_and_check_in_input,
)
from prompto.settings import Settings


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


def test_parse_file_path_and_check_in_input_error(temporary_data_folder_judge):
    # raise error if file not found
    with pytest.raises(FileNotFoundError, match="File unknown.json not found"):
        parse_file_path_and_check_in_input("unknown.json", "test")

    # raise error if file is not jsonl file path
    with pytest.raises(ValueError, match="Experiment file must be a jsonl file"):
        parse_file_path_and_check_in_input("max_queries_dict.json", "test")


def test_parse_file_path_and_check_in_input_in_input(
    temporary_data_folder_judge, caplog
):
    # case where the input file is already in the input folder
    caplog.set_level(logging.INFO)
    settings = Settings()
    result = parse_file_path_and_check_in_input(
        file_path="data/input/test-experiment.jsonl",
        settings=settings,
    )
    assert result == "test-experiment.jsonl"
    assert (
        "File data/input/test-experiment.jsonl is not in the input folder data/input"
        not in caplog.text
    )


def test_parse_file_path_and_check_in_input_not_in_input_copy(
    temporary_data_folder_judge, caplog
):
    # case where the input file is not in the input folder
    caplog.set_level(logging.INFO)
    settings = Settings()

    # should not exist in the input folder yet
    assert not os.path.isfile("data/input/test-exp-not-in-input.jsonl")

    # move_to_input is by default False
    result = parse_file_path_and_check_in_input(
        file_path="test-exp-not-in-input.jsonl",
        settings=settings,
    )
    assert result == "test-exp-not-in-input.jsonl"
    assert (
        "File test-exp-not-in-input.jsonl is not in the input folder data/input"
        in caplog.text
    )
    assert (
        "Copying file from test-exp-not-in-input.jsonl to data/input/test-exp-not-in-input.jsonl"
        in caplog.text
    )

    # should still exist in the original location
    assert os.path.isfile("test-exp-not-in-input.jsonl")
    # should be copied to the input folder
    assert os.path.isfile("data/input/test-exp-not-in-input.jsonl")


def test_parse_file_path_and_check_in_input_not_in_input_move(
    temporary_data_folder_judge, caplog
):
    # case where the input file is not in the input folder
    caplog.set_level(logging.INFO)
    settings = Settings()

    # should not exist in the input folder yet
    assert not os.path.isfile("data/input/test-exp-not-in-input.jsonl")

    result = parse_file_path_and_check_in_input(
        file_path="test-exp-not-in-input.jsonl",
        settings=settings,
        move_to_input=True,
    )
    assert result == "test-exp-not-in-input.jsonl"
    assert (
        "File test-exp-not-in-input.jsonl is not in the input folder data/input"
        in caplog.text
    )
    assert (
        "Moving file from test-exp-not-in-input.jsonl to data/input/test-exp-not-in-input.jsonl"
        in caplog.text
    )

    # should no longer exist in the original location
    assert not os.path.isfile("test-exp-not-in-input.jsonl")
    # should be copied to the input folder
    assert os.path.isfile("data/input/test-exp-not-in-input.jsonl")


def test_create_judge_experiment_judge_list(temporary_data_folder_judge):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    experiment.completed_responses = [
        {"id": 0, "prompt": "test prompt 1", "response": "test response 1"},
        {"id": 1, "prompt": "test prompt 2", "response": "test response 2"},
    ]
    js = {
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
    }
    tp = "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}"
    judge = ["judge1", "judge2"]

    assert not os.path.isfile("data/input/judge-test-experiment.jsonl")

    result = create_judge_experiment(
        create_judge_file=True,
        experiment=experiment,
        template_prompt=tp,
        judge_settings=js,
        judge=judge,
    )

    assert os.path.isfile("data/input/judge-test-experiment.jsonl")
    assert isinstance(result, Experiment)
    assert result.file_name == "judge-test-experiment.jsonl"
    assert len(result.experiment_prompts) == 4
    assert result.experiment_prompts == [
        {
            "id": "judge-judge1-0",
            "prompt": "prompt: test prompt 1 || response: test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-1",
            "prompt": "prompt: test prompt 2 || response: test response 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
        {
            "id": "judge-judge2-0",
            "prompt": "prompt: test prompt 1 || response: test response 1",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge2-1",
            "prompt": "prompt: test prompt 2 || response: test response 2",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
    ]


def test_create_judge_experiment_judge_string(temporary_data_folder_judge):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    experiment.completed_responses = [
        {"id": 0, "prompt": "test prompt 1", "response": "test response 1"},
        {"id": 1, "prompt": "test prompt 2", "response": "test response 2"},
    ]
    js = {
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
    }
    tp = "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}"
    judge = "judge1"

    assert not os.path.isfile("data/input/judge-test-experiment.jsonl")

    result = create_judge_experiment(
        create_judge_file=True,
        experiment=experiment,
        template_prompt=tp,
        judge_settings=js,
        judge=judge,
    )

    assert os.path.isfile("data/input/judge-test-experiment.jsonl")
    assert isinstance(result, Experiment)
    assert result.file_name == "judge-test-experiment.jsonl"
    assert len(result.experiment_prompts) == 2
    assert result.experiment_prompts == [
        {
            "id": "judge-judge1-0",
            "prompt": "prompt: test prompt 1 || response: test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-1",
            "prompt": "prompt: test prompt 2 || response: test response 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
    ]


def test_create_judge_experiment_type_errors(temporary_data_folder_judge):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    # add a completed response to the experiment to avoid empty error
    experiment.completed_responses = [{"prompt": "prompt1", "response": "response1"}]

    # raise error if create_judge_file is True and template_prompt is not a string
    with pytest.raises(
        TypeError,
        match="If create_judge_file is True, template_prompt must be a string",
    ):
        create_judge_experiment(
            create_judge_file=True,
            experiment=experiment,
            template_prompt=None,
            judge_settings=None,
            judge=None,
        )

    # raise error if create_judge_file is True and judge_settings is not a dictionary
    with pytest.raises(
        TypeError,
        match="If create_judge_file is True, judge_settings must be a dictionary",
    ):
        create_judge_experiment(
            create_judge_file=True,
            experiment=experiment,
            template_prompt="template",
            judge_settings=None,
            judge=None,
        )

    # raise error if create_judge_file is True and judge is not a list of strings or a string
    with pytest.raises(
        TypeError,
        match="If create_judge_file is True, judge must be a list of strings or a string",
    ):
        create_judge_experiment(
            create_judge_file=True,
            experiment=experiment,
            template_prompt="template",
            judge_settings={},
            judge=None,
        )


def test_create_judge_experiment_false(temporary_data_folder_judge):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    # add a completed response to the experiment to avoid empty error
    experiment.completed_responses = [{"prompt": "prompt1", "response": "response1"}]

    result = create_judge_experiment(
        create_judge_file=False,
        experiment=experiment,
        template_prompt=None,
        judge_settings=None,
        judge=None,
    )
    assert result is None


def test_create_judge_experiment_empty_completed_responses(temporary_data_folder_judge):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    assert experiment.completed_responses == []

    # raise error if completed_responses is empty
    with pytest.raises(
        ValueError,
        match="Cannot create judge file for experiment test-experiment as completed_responses is empty",
    ):
        create_judge_experiment(
            create_judge_file=True,
            experiment=experiment,
            template_prompt="template",
            judge_settings={},
            judge=["judge1"],
        )
