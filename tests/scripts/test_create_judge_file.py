import json
import os

from cli_test_helpers import shell

from prompto.scripts.create_judge_file import obtain_output_filepath


def test_obtain_output_filepath():
    # prepend "judge-" to the base of the experiment file name
    assert (
        obtain_output_filepath("test-experiment.jsonl", "./")
        == "./judge-test-experiment.jsonl"
    )
    assert (
        obtain_output_filepath("test-experiment.jsonl", "data/output")
        == "data/output/judge-test-experiment.jsonl"
    )
    # only take the base of the experiment file name (i.e. remove any directories)
    assert (
        obtain_output_filepath("some_directory/test-experiment.jsonl", "./")
        == "./judge-test-experiment.jsonl"
    )
    assert (
        obtain_output_filepath("some_directory/test-experiment.jsonl", "data/output")
        == "data/output/judge-test-experiment.jsonl"
    )
    # if "completed-" is in the input file name, it gets removed
    assert (
        obtain_output_filepath(
            input_filepath="some_directory/completed-test-experiment.jsonl",
            output_folder=".",
        )
        == "./judge-test-experiment.jsonl"
    )
    assert (
        obtain_output_filepath(
            input_filepath="some_directory/completed-test-experiment.jsonl",
            output_folder="data/output",
        )
        == "data/output/judge-test-experiment.jsonl"
    )


def test_create_judge_file_entrypoint():
    result = shell("prompto_create_judge_file --help")
    assert result.exit_code == 0


def test_create_judge_file_no_inputs():
    result = shell("prompto_create_judge_file")
    assert result.exit_code != 0
    assert "usage:" in result.stderr


def test_create_judge_file_input_file_not_exist(temporary_data_folder_judge):
    result = shell(
        "prompto_create_judge_file "
        "--input-file not-exist.jsonl "
        "--judge-folder judge_loc "
        "--judge judge1"
    )
    assert result.exit_code != 0
    assert (
        "FileNotFoundError: Input file 'not-exist.jsonl' is not a valid input file"
        in result.stderr
    )


def test_create_judge_file_judge_folder_not_exist(temporary_data_folder_judge):
    result = shell(
        "prompto_create_judge_file "
        "--input-file data/output/completed-test-experiment.jsonl "
        "--judge-folder not-exist-folder "
        "--judge judge1"
    )
    assert result.exit_code != 0
    assert (
        "ValueError: judge folder 'not-exist-folder' must be a valid path to a folder"
        in result.stderr
    )


def test_create_judge_file_judge_template_not_exist(temporary_data_folder_judge):
    result = shell(
        "prompto_create_judge_file "
        "--input-file data/output/completed-test-experiment.jsonl "
        "--judge-folder judge_loc_no_template "
        "--judge judge1"
    )
    assert result.exit_code != 0
    assert (
        "FileNotFoundError: Template file 'judge_loc_no_template/template.txt' does not exist"
        in result.stderr
    )


def test_create_judge_file_judge_settings_not_exist(temporary_data_folder_judge):
    result = shell(
        "prompto_create_judge_file "
        "--input-file data/output/completed-test-experiment.jsonl "
        "--judge-folder judge_loc_no_settings "
        "--judge judge1"
    )
    assert result.exit_code != 0
    assert (
        "FileNotFoundError: Judge settings file 'judge_loc_no_settings/settings.json' does not exist"
        in result.stderr
    )


def test_create_judge_file_judge_not_in_judge_settings(temporary_data_folder_judge):
    # string case
    result = shell(
        "prompto_create_judge_file "
        "--input-file data/output/completed-test-experiment.jsonl "
        "--judge-folder judge_loc "
        "--judge judge_not_in_settings"
    )
    assert result.exit_code != 0
    assert (
        "KeyError: \"Judge 'judge_not_in_settings' is not a key in judge_settings\""
        in result.stderr
    )

    # list case
    result = shell(
        "prompto_create_judge_file "
        "--input-file data/output/completed-test-experiment.jsonl "
        "--judge-folder judge_loc "
        "--judge judge1,judge_not_in_settings"
    )
    assert result.exit_code != 0
    assert (
        "KeyError: \"Judge 'judge_not_in_settings' is not a key in judge_settings\""
        in result.stderr
    )


def test_create_judge_file_full(temporary_data_folder_judge):
    result = shell(
        "prompto_create_judge_file "
        "--input-file data/output/completed-test-experiment.jsonl "
        "--judge-folder judge_loc "
        "--judge judge1 "
        "--output-folder ."
    )
    assert result.exit_code == 0
    assert os.path.isfile("./judge-test-experiment.jsonl")

    # read and check the contents of the judge file
    with open("./judge-test-experiment.jsonl", "r") as f:
        judge_inputs = [dict(json.loads(line)) for line in f]

    assert len(judge_inputs) == 3
    assert judge_inputs == [
        {
            "id": "judge-judge1-template-0",
            "template_name": "template",
            "prompt": "Template: input=test prompt 1, output=test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-template-1",
            "template_name": "template",
            "prompt": "Template: input=test prompt 2, output=test response 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
        {
            "id": "judge-judge1-template-2",
            "template_name": "template",
            "prompt": "Template: input=test prompt 3, output=test response 3",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 2,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 3",
            "input-response": "test response 3",
        },
    ]


def test_create_judge_file_full_single_templates(temporary_data_folder_judge):
    result = shell(
        "prompto_create_judge_file "
        "--input-file data/output/completed-test-experiment.jsonl "
        "--judge-folder judge_loc "
        "--judge-templates template2.txt "
        "--judge judge1 "
        "--output-folder ."
    )
    assert result.exit_code == 0
    assert os.path.isfile("./judge-test-experiment.jsonl")

    # read and check the contents of the judge file
    with open("./judge-test-experiment.jsonl", "r") as f:
        judge_inputs = [dict(json.loads(line)) for line in f]

    assert len(judge_inputs) == 3
    assert judge_inputs == [
        {
            "id": "judge-judge1-template2-0",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 1, output:test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-template2-1",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 2, output:test response 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
        {
            "id": "judge-judge1-template2-2",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 3, output:test response 3",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 2,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 3",
            "input-response": "test response 3",
        },
    ]


def test_create_judge_file_full_multiple_templates_and_judges(
    temporary_data_folder_judge,
):
    result = shell(
        "prompto_create_judge_file "
        "--input-file data/output/completed-test-experiment.jsonl "
        "--judge-folder judge_loc "
        "--judge-templates template2.txt,template.txt "
        "--judge judge1,judge2 "
        "--output-folder ."
    )
    assert result.exit_code == 0
    assert os.path.isfile("./judge-test-experiment.jsonl")

    # read and check the contents of the judge file
    with open("./judge-test-experiment.jsonl", "r") as f:
        judge_inputs = [dict(json.loads(line)) for line in f]

    assert len(judge_inputs) == 12
    assert judge_inputs == [
        {
            "id": "judge-judge1-template2-0",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 1, output:test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-template2-1",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 2, output:test response 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
        {
            "id": "judge-judge1-template2-2",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 3, output:test response 3",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 2,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 3",
            "input-response": "test response 3",
        },
        {
            "id": "judge-judge1-template-0",
            "template_name": "template",
            "prompt": "Template: input=test prompt 1, output=test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-template-1",
            "template_name": "template",
            "prompt": "Template: input=test prompt 2, output=test response 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
        {
            "id": "judge-judge1-template-2",
            "template_name": "template",
            "prompt": "Template: input=test prompt 3, output=test response 3",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 2,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 3",
            "input-response": "test response 3",
        },
        {
            "id": "judge-judge2-template2-0",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 1, output:test response 1",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 0,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge2-template2-1",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 2, output:test response 2",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 1,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
        {
            "id": "judge-judge2-template2-2",
            "template_name": "template2",
            "prompt": "Template 2: input:test prompt 3, output:test response 3",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 2,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 3",
            "input-response": "test response 3",
        },
        {
            "id": "judge-judge2-template-0",
            "template_name": "template",
            "prompt": "Template: input=test prompt 1, output=test response 1",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 0,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge2-template-1",
            "template_name": "template",
            "prompt": "Template: input=test prompt 2, output=test response 2",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 1,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
        {
            "id": "judge-judge2-template-2",
            "template_name": "template",
            "prompt": "Template: input=test prompt 3, output=test response 3",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 2,
            "input-api": "test",
            "input-model": "test_model",
            "input-prompt": "test prompt 3",
            "input-response": "test response 3",
        },
    ]
