import logging
import os

import pytest

from prompto.experiment import Experiment
from prompto.rephrasal import Rephraser
from prompto.scripts.run_experiment import (
    create_judge_experiment,
    create_rephrase_experiment,
    load_env_file,
    load_judge_args,
    load_max_queries_json,
    load_rephrase_args,
    parse_file_path_and_check_in_input,
)
from prompto.settings import Settings

COMPLETED_RESPONSES = [
    {"id": 0, "prompt": "test prompt 1", "response": "test response 1"},
    {"id": 1, "prompt": "test prompt 2", "response": "test response 2"},
]
JUDGE_SETTINGS = {
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
REPHRASE_SETTINGS = {
    "rephrase1": {
        "api": "test",
        "model_name": "model1",
        "parameters": {"temperature": 0.5},
    },
    "rephrase2": {
        "api": "test",
        "model_name": "model2",
        "parameters": {"temperature": 0.2, "top_k": 0.9},
    },
}


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


def test_parse_file_path_and_check_in_input_error(temporary_data_folder_judge):
    # raise error if file not found
    with pytest.raises(FileNotFoundError, match="File unknown.json not found"):
        parse_file_path_and_check_in_input("unknown.json", "test")

    # raise error if file is not jsonl file path
    with pytest.raises(ValueError, match="Experiment file must be a jsonl or csv file"):
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


def test_load_rephrase_args_all_none(temporary_data_folder_rephrase, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_rephrase_args(
        rephrase_folder_arg=None, rephrase_model_arg=None, rephrase_templates_arg=None
    )
    assert result == (False, None, None, None)
    assert (
        "Not creating rephrase file as one of rephrase_folder, rephrase or templates is None"
        in caplog.text
    )


def test_load_rephrase_args_rephrase_arg_none(temporary_data_folder_rephrase, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_rephrase_args(
        rephrase_folder_arg="rephrase_loc",
        rephrase_model_arg=None,
        rephrase_templates_arg="template.txt",
    )
    assert result == (False, None, None, None)
    assert (
        "Not creating rephrase file as one of rephrase_folder, rephrase or templates is None"
        in caplog.text
    )


def test_load_rephrase_args_rephrase_folder_arg_none(
    temporary_data_folder_rephrase, caplog
):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_rephrase_args(
        rephrase_folder_arg=None,
        rephrase_model_arg="rephrase1",
        rephrase_templates_arg="template.txt",
    )
    assert result == (False, None, None, None)
    assert (
        "Not creating rephrase file as one of rephrase_folder, rephrase or templates is None"
        in caplog.text
    )


def test_load_rephrase_args_templates_arg_none(temporary_data_folder_rephrase, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_rephrase_args(
        rephrase_folder_arg="rephrase_loc",
        rephrase_model_arg="rephrase1",
        rephrase_templates_arg=None,
    )
    assert result == (False, None, None, None)
    assert (
        "Not creating rephrase file as one of rephrase_folder, rephrase or templates is None"
        in caplog.text
    )


def test_load_rephrase_args(temporary_data_folder_rephrase, caplog):
    caplog.set_level(logging.INFO)
    # if both arguments are not None, return (True, templaate, rephrase_settings, rephrase
    result = load_rephrase_args(
        rephrase_folder_arg="rephrase_loc",
        rephrase_model_arg="rephrase1,rephrase2",
        rephrase_templates_arg="template.txt",
    )
    assert result == (
        True,
        ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"],
        {
            "rephrase1": {
                "api": "test",
                "model_name": "model1",
                "parameters": {"temperature": 0.5},
            },
            "rephrase2": {
                "api": "test",
                "model_name": "model2",
                "parameters": {"temperature": 0.2, "top_k": 0.9},
            },
        },
        ["rephrase1", "rephrase2"],
    )


def test_create_rephrase_experiment_rephrase_model_list(temporary_data_folder_rephrase):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase_model = ["rephrase1", "rephrase2"]

    assert not os.path.isfile("data/input/rephrase-test-experiment.jsonl")

    result_experiment, rephraser = create_rephrase_experiment(
        create_rephrase_file=True,
        experiment=experiment,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
        rephrase_model=rephrase_model,
    )

    assert os.path.isfile("data/input/rephrase-test-experiment.jsonl")
    assert isinstance(result_experiment, Experiment)
    assert result_experiment.file_name == "rephrase-test-experiment.jsonl"
    expected_result = [
        {
            "id": "rephrase-rephrase1-0-0",
            "template_index": 0,
            "prompt": "Template 1: test prompt 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-api": "test",
            "input-model_name": "model1",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "This is a test response",
        },
        {
            "id": "rephrase-rephrase1-0-1",
            "template_index": 0,
            "prompt": "Template 1: test prompt 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-api": "test",
            "input-model_name": "model2",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "something else",
        },
        {
            "id": "rephrase-rephrase1-1-0",
            "template_index": 1,
            "prompt": "Template 2: \ntest prompt 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-api": "test",
            "input-model_name": "model1",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "This is a test response",
        },
        {
            "id": "rephrase-rephrase1-1-1",
            "template_index": 1,
            "prompt": "Template 2: \ntest prompt 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-api": "test",
            "input-model_name": "model2",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "something else",
        },
        {
            "id": "rephrase-rephrase2-0-0",
            "template_index": 0,
            "prompt": "Template 1: test prompt 1",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-api": "test",
            "input-model_name": "model1",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "This is a test response",
        },
        {
            "id": "rephrase-rephrase2-0-1",
            "template_index": 0,
            "prompt": "Template 1: test prompt 2",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-api": "test",
            "input-model_name": "model2",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "something else",
        },
        {
            "id": "rephrase-rephrase2-1-0",
            "template_index": 1,
            "prompt": "Template 2: \ntest prompt 1",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-api": "test",
            "input-model_name": "model1",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "This is a test response",
        },
        {
            "id": "rephrase-rephrase2-1-1",
            "template_index": 1,
            "prompt": "Template 2: \ntest prompt 2",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-api": "test",
            "input-model_name": "model2",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "something else",
        },
    ]

    assert len(result_experiment.experiment_prompts) == 8
    assert result_experiment.experiment_prompts == expected_result
    assert isinstance(rephraser, Rephraser)
    assert rephraser.rephrase_prompts == expected_result


def test_create_rephrase_experiment_rephrase_model_string(
    temporary_data_folder_rephrase,
):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase_model = "rephrase1"

    assert not os.path.isfile("data/input/rephrase-test-experiment.jsonl")

    result_experiment, rephraser = create_rephrase_experiment(
        create_rephrase_file=True,
        experiment=experiment,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
        rephrase_model=rephrase_model,
    )

    assert os.path.isfile("data/input/rephrase-test-experiment.jsonl")
    assert isinstance(result_experiment, Experiment)
    assert result_experiment.file_name == "rephrase-test-experiment.jsonl"
    expected_result = [
        {
            "id": "rephrase-rephrase1-0-0",
            "template_index": 0,
            "prompt": "Template 1: test prompt 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-api": "test",
            "input-model_name": "model1",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "This is a test response",
        },
        {
            "id": "rephrase-rephrase1-0-1",
            "template_index": 0,
            "prompt": "Template 1: test prompt 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-api": "test",
            "input-model_name": "model2",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "something else",
        },
        {
            "id": "rephrase-rephrase1-1-0",
            "template_index": 1,
            "prompt": "Template 2: \ntest prompt 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-api": "test",
            "input-model_name": "model1",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "This is a test response",
        },
        {
            "id": "rephrase-rephrase1-1-1",
            "template_index": 1,
            "prompt": "Template 2: \ntest prompt 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-api": "test",
            "input-model_name": "model2",
            "input-parameters": {"raise_error": "False"},
            "input-expected_response": "something else",
        },
    ]

    assert len(result_experiment.experiment_prompts) == 4
    assert result_experiment.experiment_prompts == expected_result
    assert isinstance(rephraser, Rephraser)
    assert rephraser.rephrase_prompts == expected_result


def test_create_rephrase_experiment_type_errors(temporary_data_folder_rephrase):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    # add a completed response to the experiment to avoid empty error
    experiment.completed_responses = [{"prompt": "prompt1", "response": "response1"}]

    # raise error if create_rephrase_file is True and template_prompts is not a dictionary
    with pytest.raises(
        TypeError,
        match="If create_rephrase_file is True, template_prompts must be a list of strings",
    ):
        create_rephrase_experiment(
            create_rephrase_file=True,
            experiment=experiment,
            template_prompts=None,
            rephrase_settings=None,
            rephrase_model=None,
        )

    # raise error if create_rephrase_file is True and REPHRASE_SETTINGS is not a dictionary
    with pytest.raises(
        TypeError,
        match="If create_rephrase_file is True, rephrase_settings must be a dictionary",
    ):
        create_rephrase_experiment(
            create_rephrase_file=True,
            experiment=experiment,
            template_prompts=["some template"],
            rephrase_settings=None,
            rephrase_model=None,
        )

    # raise error if create_rephrase_file is True and rephrase_model is not a list of strings or a string
    with pytest.raises(
        TypeError,
        match="If create_rephrase_file is True, rephrase_model must be a list of strings or a string",
    ):
        create_rephrase_experiment(
            create_rephrase_file=True,
            experiment=experiment,
            template_prompts=["some template"],
            rephrase_settings={},
            rephrase_model=None,
        )


def test_create_rephrase_experiment_false(temporary_data_folder_rephrase):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    experiment.completed_responses = [{"prompt": "prompt1", "response": "response1"}]

    result_experiment, rephraser = create_rephrase_experiment(
        create_rephrase_file=False,
        experiment=experiment,
        template_prompts=None,
        rephrase_settings=None,
        rephrase_model=None,
    )
    assert result_experiment is None
    assert rephraser is None


def test_load_judge_args_all_none(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_judge_args(
        judge_folder_arg=None, judge_arg=None, judge_templates_arg=None
    )
    assert result == (False, None, None, None)
    assert (
        "Not creating judge file as one of judge_folder, judge or templates is None"
        in caplog.text
    )


def test_load_judge_args_judge_arg_none(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_judge_args(
        judge_folder_arg="judge_loc", judge_arg=None, judge_templates_arg="template.txt"
    )
    assert result == (False, None, None, None)
    assert (
        "Not creating judge file as one of judge_folder, judge or templates is None"
        in caplog.text
    )


def test_load_judge_args_judge_folder_arg_none(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_judge_args(
        judge_folder_arg=None, judge_arg="judge1", judge_templates_arg="template.txt"
    )
    assert result == (False, None, None, None)
    assert (
        "Not creating judge file as one of judge_folder, judge or templates is None"
        in caplog.text
    )


def test_load_judge_args_templates_arg_none(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if either argument is None, return (False, None, None, None)
    result = load_judge_args(
        judge_folder_arg="judge_loc", judge_arg="judge1", judge_templates_arg=None
    )
    assert result == (False, None, None, None)
    assert (
        "Not creating judge file as one of judge_folder, judge or templates is None"
        in caplog.text
    )


def test_load_judge_args(temporary_data_folder_judge, caplog):
    caplog.set_level(logging.INFO)
    # if both arguments are not None, return (True, templaate, judge_settings, judge
    result = load_judge_args(
        judge_folder_arg="judge_loc",
        judge_arg="judge1,judge2",
        judge_templates_arg="template.txt",
    )
    assert result == (
        True,
        {"template": "Template: input={INPUT_PROMPT}, output={OUTPUT_RESPONSE}"},
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


def test_create_judge_experiment_judge_list(temporary_data_folder_judge):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    experiment.completed_responses = COMPLETED_RESPONSES
    tp = {"temp": "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}"}
    judge = ["judge1", "judge2"]

    assert not os.path.isfile("data/input/judge-test-experiment.jsonl")

    result = create_judge_experiment(
        create_judge_file=True,
        experiment=experiment,
        template_prompts=tp,
        judge_settings=JUDGE_SETTINGS,
        judge=judge,
    )

    assert os.path.isfile("data/input/judge-test-experiment.jsonl")
    assert isinstance(result, Experiment)
    assert result.file_name == "judge-test-experiment.jsonl"
    assert len(result.experiment_prompts) == 4
    assert result.experiment_prompts == [
        {
            "id": "judge-judge1-temp-0",
            "template_name": "temp",
            "prompt": "prompt: test prompt 1 || response: test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-temp-1",
            "template_name": "temp",
            "prompt": "prompt: test prompt 2 || response: test response 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
        {
            "id": "judge-judge2-temp-0",
            "template_name": "temp",
            "prompt": "prompt: test prompt 1 || response: test response 1",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge2-temp-1",
            "template_name": "temp",
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
    experiment.completed_responses = COMPLETED_RESPONSES
    tp = {"temp": "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}"}
    judge = "judge1"

    assert not os.path.isfile("data/input/judge-test-experiment.jsonl")

    result = create_judge_experiment(
        create_judge_file=True,
        experiment=experiment,
        template_prompts=tp,
        judge_settings=JUDGE_SETTINGS,
        judge=judge,
    )

    assert os.path.isfile("data/input/judge-test-experiment.jsonl")
    assert isinstance(result, Experiment)
    assert result.file_name == "judge-test-experiment.jsonl"
    assert len(result.experiment_prompts) == 2
    assert result.experiment_prompts == [
        {
            "id": "judge-judge1-temp-0",
            "template_name": "temp",
            "prompt": "prompt: test prompt 1 || response: test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-temp-1",
            "template_name": "temp",
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

    # raise error if create_judge_file is True and template_prompts is not a dictionary
    with pytest.raises(
        TypeError,
        match="If create_judge_file is True, template_prompts must be a dictionary",
    ):
        create_judge_experiment(
            create_judge_file=True,
            experiment=experiment,
            template_prompts=None,
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
            template_prompts={"template": "some template"},
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
            template_prompts={"template": "some template"},
            judge_settings={},
            judge=None,
        )


def test_create_judge_experiment_false(temporary_data_folder_judge):
    settings = Settings()
    experiment = Experiment("test-experiment.jsonl", settings)
    experiment.completed_responses = [{"prompt": "prompt1", "response": "response1"}]

    result = create_judge_experiment(
        create_judge_file=False,
        experiment=experiment,
        template_prompts=None,
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
            template_prompts={"template": "some template"},
            judge_settings={},
            judge=["judge1"],
        )
