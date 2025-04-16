import json
import os

import pytest

from prompto.judge import Judge, load_judge_folder

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


def test_load_judge_folder(temporary_data_folder_judge):
    # test the function reads template.txt and settings.json correctly
    template_prompt, judge_settings = load_judge_folder("judge_loc")
    assert template_prompt == {
        "template": "Template: input={INPUT_PROMPT}, output={OUTPUT_RESPONSE}"
    }
    assert judge_settings == {
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


def test_load_judge_folder_string_as_arg(temporary_data_folder_judge):
    # test the function reads template.txt and settings.json correctly
    template_prompt, judge_settings = load_judge_folder(
        "judge_loc", templates="template2.txt"
    )
    assert template_prompt == {
        "template2": "Template 2: input:{INPUT_PROMPT}, output:{OUTPUT_RESPONSE}"
    }
    assert judge_settings == {
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


def test_load_judge_folder_multiple_templates(temporary_data_folder_judge):
    # test the function reads template.txt and settings.json correctly
    template_prompt, judge_settings = load_judge_folder(
        "judge_loc", templates=["template.txt", "template2.txt"]
    )
    assert template_prompt == {
        "template": "Template: input={INPUT_PROMPT}, output={OUTPUT_RESPONSE}",
        "template2": "Template 2: input:{INPUT_PROMPT}, output:{OUTPUT_RESPONSE}",
    }
    assert judge_settings == {
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


def test_load_judge_folder_arg_error(temporary_data_folder_judge):
    # raise error if judge folder is not a valid path to a directory
    with pytest.raises(
        ValueError,
        match="judge folder 'non_existent_folder' must be a valid path to a folder",
    ):
        load_judge_folder("non_existent_folder")

    # raise error if template file does not exist in the judge folder
    # default template.txt case
    with pytest.raises(
        FileNotFoundError,
        match="Template file 'judge_loc_no_template/template.txt' does not exist",
    ):
        load_judge_folder("judge_loc_no_template")

    # string template case
    with pytest.raises(
        FileNotFoundError,
        match="Template file 'judge_loc/some-other-template.txt' does not exist",
    ):
        load_judge_folder("judge_loc", templates="some-other-template.txt")

    # list of templates case
    with pytest.raises(
        FileNotFoundError,
        match="Template file 'judge_loc/some-other-template.txt' does not exist",
    ):
        load_judge_folder(
            "judge_loc", templates=["template.txt", "some-other-template.txt"]
        )

    # raise error if template file is not a .txt file
    with pytest.raises(
        ValueError,
        match="Template file 'judge_loc/template.json' must end with '.txt'",
    ):
        load_judge_folder("judge_loc", templates="template.json")

    # raise error if settings file does not exist in the judge folder
    with pytest.raises(
        FileNotFoundError,
        match="Judge settings file 'judge_loc_no_settings/settings.json' does not exist",
    ):
        load_judge_folder("judge_loc_no_settings")


def test_judge_check_judge_settings():
    # raise error if nothing is provided
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'judge_settings'"
    ):
        Judge.check_judge_settings()

    # raise error if judge settings is not a dictionary
    with pytest.raises(
        TypeError,
        match="judge_settings must be a dictionary",
    ):
        Judge.check_judge_settings("not_a_dict")

    # raise error if a value is not a dictionary
    with pytest.raises(
        TypeError,
        match="Value for judge key 'judge1' must be a dictionary",
    ):
        Judge.check_judge_settings({"judge1": "not_a_dict"})

    # raise error if a judge settings has no "api" key
    with pytest.raises(
        KeyError,
        match="'api' key not found in settings for judge 'judge'",
    ):
        Judge.check_judge_settings(
            {"judge": {"model_name": "model1", "parameters": {"temperature": 0.5}}}
        )

    # raise error if judge settings has no "model_name" key
    with pytest.raises(
        KeyError,
        match="'model_name' key not found in settings for judge 'judge'",
    ):
        Judge.check_judge_settings(
            {"judge": {"api": "test", "parameters": {"temperature": 0.5}}}
        )

    # raise error if judge settings has no "parameters" key
    with pytest.raises(
        KeyError,
        match="'parameters' key not found in settings for judge 'judge'",
    ):
        Judge.check_judge_settings({"judge": {"api": "test", "model_name": "model1"}})

    # raise error if parameters is not a dictionary
    with pytest.raises(
        TypeError,
        match="Value for 'parameters' key must be a dictionary for judge 'judge'",
    ):
        Judge.check_judge_settings(
            {
                "judge": {
                    "api": "test",
                    "model_name": "model1",
                    "parameters": "not_a_dict",
                }
            }
        )

    # passes
    assert Judge.check_judge_settings(
        {
            "judge": {
                "api": "test",
                "model_name": "model1",
                "parameters": {"temperature": 0.5},
            }
        }
    )
    assert Judge.check_judge_settings(
        {
            "judge1": {
                "api": "test",
                "model_name": "model1",
                "parameters": {"temperature": 0.5},
            },
            "judge2": {
                "api": "test",
                "model_name": "model2",
                "parameters": {"temperature": 0.5},
            },
        }
    )


def test_check_judge_in_judge_settings():
    # raise error if judge is not a key in judge settings
    # judge is a string case
    with pytest.raises(
        KeyError,
        match="Judge 'judge' is not a key in judge_settings",
    ):
        Judge.check_judge_in_judge_settings("judge", {"judge1": {}, "judge2": {}})

    # judge is a list of strings case (of one string)
    with pytest.raises(
        KeyError,
        match="Judge 'judge' is not a key in judge_settings",
    ):
        Judge.check_judge_in_judge_settings(["judge"], {"judge1": {}, "judge2": {}})

    # judge is a list of strings case (of multiple strings)
    with pytest.raises(
        KeyError,
        match="Judge 'judge' is not a key in judge_settings",
    ):
        Judge.check_judge_in_judge_settings(
            ["judge1", "judge"], {"judge1": {}, "judge2": {}}
        )

    # judge is a list but some are not strings
    with pytest.raises(
        TypeError,
        match="If judge is a list, each element must be a string",
    ):
        Judge.check_judge_in_judge_settings(["judge1", 2], {"judge1": {}, "judge2": {}})

    # passes
    assert Judge.check_judge_in_judge_settings(
        judge="judge1", judge_settings={"judge1": {}, "judge2": {}}
    )
    assert Judge.check_judge_in_judge_settings(
        judge=["judge1"], judge_settings={"judge1": {}, "judge2": {}}
    )
    assert Judge.check_judge_in_judge_settings(
        judge=["judge1", "judge2"], judge_settings={"judge1": {}, "judge2": {}}
    )


def test_check_judge_init():
    # raise error if nothing is provided
    with pytest.raises(
        TypeError,
        match="missing 3 required positional arguments",
    ):
        Judge()

    # raise error if template_prompts is not a dictionary
    with pytest.raises(
        TypeError,
        match="template_prompts must be a dictionary",
    ):
        Judge(
            completed_responses="completed_responses (no check on list of dicts)",
            template_prompts="not_a_dict",
            judge_settings=JUDGE_SETTINGS,
        )

    # raise error if judge_settings is not a valid dictionary
    with pytest.raises(
        TypeError,
        match="judge_settings must be a dictionary",
    ):
        Judge(
            completed_responses="completed_responses (no check on list of dicts)",
            template_prompts={"template": "some template"},
            judge_settings="not_a_dict",
        )

    tp = {"temp": "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}"}
    judge = Judge(
        completed_responses=COMPLETED_RESPONSES,
        template_prompts=tp,
        judge_settings=JUDGE_SETTINGS,
    )
    assert judge.completed_responses == COMPLETED_RESPONSES
    assert judge.judge_settings == JUDGE_SETTINGS
    assert judge.template_prompts == tp
    assert judge.judge_prompts == []


def test_judge_create_judge_inputs_errors():
    tp = {"temp": "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}"}
    judge = Judge(
        completed_responses=COMPLETED_RESPONSES,
        template_prompts=tp,
        judge_settings=JUDGE_SETTINGS,
    )

    # raise error if judge not provided
    with pytest.raises(
        TypeError,
        match="missing 1 required positional argument: 'judge'",
    ):
        judge.create_judge_inputs()

    # raise error if judge is not in judge settings
    with pytest.raises(
        KeyError,
        match="Judge 'judge' is not a key in judge_settings",
    ):
        judge.create_judge_inputs("judge")

    # raise error if judge is not in judge settings (list case)
    with pytest.raises(
        KeyError,
        match="Judge 'judge' is not a key in judge_settings",
    ):
        judge.create_judge_inputs(["judge", "judge1"])

    # raise error if judge is a list but some are not strings
    with pytest.raises(
        TypeError,
        match="If judge is a list, each element must be a string",
    ):
        judge.create_judge_inputs(["judge1", 2])


def test_judge_create_judge_inputs(capsys):
    tp = {"temp": "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}"}
    judge = Judge(
        completed_responses=COMPLETED_RESPONSES,
        template_prompts=tp,
        judge_settings=JUDGE_SETTINGS,
    )

    # "judge1" case
    judge_1_inputs = judge.create_judge_inputs("judge1")
    expected_result = [
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
    assert len(judge_1_inputs) == 2
    assert judge_1_inputs == expected_result
    assert judge.judge_prompts == expected_result

    captured = capsys.readouterr()
    assert (
        "Creating judge inputs for judge 'judge1' and template 'temp'" in captured.err
    )

    # "judge2" case
    judge_2_inputs = judge.create_judge_inputs("judge2")
    expected_result_2 = [
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
    assert len(judge_2_inputs) == 2
    assert judge_2_inputs == expected_result_2
    assert judge.judge_prompts == expected_result_2

    captured = capsys.readouterr()
    assert (
        "Creating judge inputs for judge 'judge2' and template 'temp'" in captured.err
    )


def test_judge_create_judge_inputs_multiple_judges(capsys):
    tp = {"temp": "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}"}
    judge = Judge(
        completed_responses=COMPLETED_RESPONSES,
        template_prompts=tp,
        judge_settings=JUDGE_SETTINGS,
    )

    # "judge1, judge2" case
    judge_1_2_inputs = judge.create_judge_inputs(["judge1", "judge2"])
    expected_result = [
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
    assert len(judge_1_2_inputs) == 4
    assert judge_1_2_inputs == expected_result
    assert judge.judge_prompts == expected_result

    captured = capsys.readouterr()
    assert (
        "Creating judge inputs for judge 'judge1' and template 'temp'" in captured.err
    )
    assert (
        "Creating judge inputs for judge 'judge2' and template 'temp'" in captured.err
    )


def test_judge_create_judge_file(temporary_data_folder_judge, capsys):
    # case where template_prompt has multiple templates
    tp = {
        "temp": "prompt: {INPUT_PROMPT} || response: {OUTPUT_RESPONSE}",
        "temp2": "prompt 2: {INPUT_PROMPT} || response 2: {OUTPUT_RESPONSE}",
    }
    judge = Judge(
        completed_responses=COMPLETED_RESPONSES,
        template_prompts=tp,
        judge_settings=JUDGE_SETTINGS,
    )

    # raise error if nothing is provided
    with pytest.raises(
        TypeError,
        match="missing 2 required positional arguments",
    ):
        judge.create_judge_file()

    # raise error if out_filepath is not a string that ends with ".jsonl"
    with pytest.raises(
        ValueError,
        match="out_filepath must end with '.jsonl'",
    ):
        judge.create_judge_file(judge="judge", out_filepath="judge_file")

    # create judge file
    judge.create_judge_file(judge="judge1", out_filepath="judge_file.jsonl")

    captured = capsys.readouterr()
    assert (
        "Creating judge inputs for judge 'judge1' and template 'temp'" in captured.err
    )
    assert (
        "Creating judge inputs for judge 'judge1' and template 'temp2'" in captured.err
    )
    assert "Writing judge prompts to judge_file.jsonl" in captured.err

    # check the judge file was created
    assert os.path.isfile("judge_file.jsonl")

    # read and check the contents of the judge file
    with open("judge_file.jsonl", "r") as f:
        judge_inputs = [dict(json.loads(line)) for line in f]

    expected_result = [
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
            "id": "judge-judge1-temp2-0",
            "template_name": "temp2",
            "prompt": "prompt 2: test prompt 1 || response 2: test response 1",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-response": "test response 1",
        },
        {
            "id": "judge-judge1-temp2-1",
            "template_name": "temp2",
            "prompt": "prompt 2: test prompt 2 || response 2: test response 2",
            "api": "test",
            "model_name": "model1",
            "parameters": {"temperature": 0.5},
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "input-response": "test response 2",
        },
    ]

    assert len(judge_inputs) == 4
    assert judge_inputs == expected_result
    assert judge.judge_prompts == expected_result
