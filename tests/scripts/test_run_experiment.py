import json
import logging
import os

import pytest
from cli_test_helpers import shell

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

    assert os.path.isfile("test-exp-not-in-input.jsonl")
    assert os.path.isfile("data/input/test-exp-not-in-input.jsonl")


def test_parse_file_path_and_check_in_input_not_in_input_move(
    temporary_data_folder_judge, caplog
):
    # case where the input file is not in the input folder
    caplog.set_level(logging.INFO)
    settings = Settings()

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

    assert not os.path.isfile("test-exp-not-in-input.jsonl")
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


def test_run_experiment_entrypoint():
    result = shell("prompto_run_experiment --help")
    assert result.exit_code == 0


def test_run_experiment_no_inputs():
    result = shell("prompto_run_experiment")
    assert result.exit_code != 0
    assert "usage:" in result.stderr


def test_run_experiment_no_judge_in_input(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert (
        "Not creating judge file as one of judge_location or judge is None"
        in result.stderr
    )
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.jsonl..."
        in result.stderr
    )
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")


def test_run_experiment_no_judge_not_in_input_move(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file test-exp-not-in-input.jsonl "
        "--env-file some_file.env "
        "--data-folder pipeline_data "
        "--move-to-input "
        "--max-queries 500 "
        "--max-attempts 10 "
        "--parallel "
        "--max-queries-json max_queries_dict.json"
    )
    assert result.exit_code == 0

    assert not os.path.isfile("test-exp-not-in-input.jsonl")

    assert "No environment file found at some_file.env" in result.stderr
    assert (
        "Not creating judge file as one of judge_location or judge is None"
        in result.stderr
    )
    assert (
        "File test-exp-not-in-input.jsonl is not in the input folder pipeline_data/input"
        in result.stderr
    )
    assert (
        "Moving file from test-exp-not-in-input.jsonl to pipeline_data/input/test-exp-not-in-input.jsonl"
        in result.stderr
    )
    assert (
        "Settings: "
        "data_folder=pipeline_data, "
        "max_queries=500, "
        "max_attempts=10, "
        "parallel=True, "
        "max_queries_dict={'test': {'model1': 100, 'model2': 120}}\n"
        "Subfolders: "
        "input_folder=pipeline_data/input, "
        "output_folder=pipeline_data/output, "
        "media_folder=pipeline_data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: test-exp-not-in-input.jsonl..."
        in result.stderr
    )
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("pipeline_data/output/test-exp-not-in-input")


def test_run_experiment_judge_not_in_input_copy(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file test-exp-not-in-input.jsonl "
        "--data-folder pipeline_data "
        "--max-queries=200 "
        "--judge-location judge_loc "
        "--judge judge1"
    )
    assert result.exit_code == 0

    assert os.path.isfile("test-exp-not-in-input.jsonl")

    assert "No environment file found at .env" in result.stderr
    assert "Judge location loaded from judge_loc" in result.stderr
    assert "Judges to be used: ['judge1']" in result.stderr
    assert (
        "File test-exp-not-in-input.jsonl is not in the input folder pipeline_data/input"
        in result.stderr
    )
    assert (
        "Copying file from test-exp-not-in-input.jsonl to pipeline_data/input/test-exp-not-in-input.jsonl"
        in result.stderr
    )
    assert (
        "Settings: "
        "data_folder=pipeline_data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=pipeline_data/input, "
        "output_folder=pipeline_data/output, "
        "media_folder=pipeline_data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: test-exp-not-in-input.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: test-exp-not-in-input.jsonl" in result.stderr
    assert (
        "Starting processing judge of experiment: judge-test-exp-not-in-input.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: judge-test-exp-not-in-input.jsonl" in result.stderr
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("pipeline_data/output/test-exp-not-in-input")
    assert os.path.isdir("pipeline_data/output/judge-test-exp-not-in-input")


def test_run_experiment_judge(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200 "
        "--judge-location judge_loc "
        "--judge judge1,judge2"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert "Judge location loaded from judge_loc" in result.stderr
    assert "Judges to be used: ['judge1', 'judge2']" in result.stderr
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: test-experiment.jsonl" in result.stderr
    assert (
        "Starting processing judge of experiment: judge-test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: judge-test-experiment.jsonl" in result.stderr
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")
    assert os.path.isdir("data/output/judge-test-experiment")


def test_run_experiment_scorer_not_in_dict(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200 "
        "--scorer not_a_scorer"
    )
    assert result.exit_code != 0
    assert (
        "Scorer 'not_a_scorer' is not a key in scoring_functions_dict. "
        "Available scorers are: "
    ) in result.stderr


def test_run_experiment_scorer(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200 "
        "--scorer match,includes"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert (
        "Not creating judge file as one of judge_location or judge is None"
        in result.stderr
    )
    assert "Scoring functions to be used: ['match', 'includes']" in result.stderr
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.jsonl..."
        in result.stderr
    )
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")

    completed_files = [
        x for x in os.listdir("data/output/test-experiment") if "completed" in x
    ]
    assert len(completed_files) == 1
    completed_file = completed_files[0]

    # load the output to check the scores have been added
    with open(f"data/output/test-experiment/{completed_file}", "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 2
    assert responses[0]["match"] is True
    assert responses[1]["match"] is False
    assert responses[0]["includes"] is True
    assert responses[1]["includes"] is False


def test_run_experiment_judge_and_scorer(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200 "
        "--judge-location judge_loc "
        "--judge judge2 "
        "--scorer 'match, includes'"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert "Judge location loaded from judge_loc" in result.stderr
    assert "Judges to be used: ['judge2']" in result.stderr
    assert "Scoring functions to be used: ['match', 'includes']" in result.stderr
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: test-experiment.jsonl" in result.stderr
    assert (
        "Starting processing judge of experiment: judge-test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: judge-test-experiment.jsonl" in result.stderr
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")
    assert os.path.isdir("data/output/judge-test-experiment")

    # check the output files for the test-experiment
    completed_files = [
        x for x in os.listdir("data/output/test-experiment") if "completed" in x
    ]
    assert len(completed_files) == 1
    completed_file = completed_files[0]

    # load the output to check the scores have been added
    with open(f"data/output/test-experiment/{completed_file}", "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 2
    assert responses[0]["match"] is True
    assert responses[1]["match"] is False
    assert responses[0]["includes"] is True
    assert responses[1]["includes"] is False

    # check the output files for the judge-test-experiment
    completed_files = [
        x for x in os.listdir("data/output/judge-test-experiment") if "completed" in x
    ]
    assert len(completed_files) == 1
    completed_file = completed_files[0]

    # load the output to check the scores have been added
    with open(f"data/output/judge-test-experiment/{completed_file}", "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 2
    assert responses[0]["input-match"] is True
    assert responses[1]["input-match"] is False
    assert responses[0]["input-includes"] is True
    assert responses[1]["input-includes"] is False
