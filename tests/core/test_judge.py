import json
import logging
import os

import pytest

from prompto.judge import Judge, parse_judge_arg, parse_judge_location_arg


def test_parse_judge_arg():
    # test parser of judge argument
    assert parse_judge_arg("judge") == ["judge"]
    assert parse_judge_arg("judge1,judge2") == ["judge1", "judge2"]
    assert parse_judge_arg("judge1, judge2") == ["judge1", "judge2"]
    assert parse_judge_arg(" judge1 , judge2") == ["judge1", "judge2"]
    assert parse_judge_arg("judge1,judge2,judge1") == ["judge1", "judge2"]
    assert parse_judge_arg("judge1,judge2 , judge1") == ["judge1", "judge2"]
    assert parse_judge_arg("judge1,judge2,judge2") == ["judge1", "judge2"]
    assert parse_judge_arg("judge2,judge2,judge1,judge3") == [
        "judge2",
        "judge1",
        "judge3",
    ]


def test_parse_judge_arg_logging(caplog):
    # test logging information is correct
    caplog.set_level(logging.INFO)

    assert parse_judge_arg("judge1, judge2") == ["judge1", "judge2"]
    assert "Judges to be used: ['judge1', 'judge2']" in caplog.text

    assert parse_judge_arg("judge_3,judge_4,judge5") == ["judge_3", "judge_4", "judge5"]
    assert "Judges to be used: ['judge_3', 'judge_4', 'judge5']" in caplog.text


def test_parse_judge_location_arg(temporary_data_folder_judge):
    # test the function reads template.txt and settings.json correctly
    template_prompt, judge_settings = parse_judge_location_arg("judge_loc")
    assert template_prompt == (
        "Template: input={INPUT_PROMPT}, output={OUTPUT_RESPONSE}"
    )
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


def test_parse_location_arg_error(temporary_data_folder_judge):
    # raise error if judge location is not a valid path to a directory
    with pytest.raises(
        ValueError,
        match="Judge location 'non_existent_folder' must be a valid path to a folder",
    ):
        parse_judge_location_arg("non_existent_folder")

    # raise error if template file does not exist in the judge location
    with pytest.raises(
        FileNotFoundError,
        match="Template file 'judge_loc_no_template/template.txt' does not exist",
    ):
        parse_judge_location_arg("judge_loc_no_template")

    # raise error if settings file does not exist in the judge location
    with pytest.raises(
        FileNotFoundError,
        match="Judge settings file 'judge_loc_no_settings/settings.json' does not exist",
    ):
        parse_judge_location_arg("judge_loc_no_settings")


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


def check_judge_in_judge_settings():
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


def check_judge_init():
    # raise error if nothing is provided
    with pytest.raises(
        TypeError,
        match="missing 3 required positional arguments",
    ):
        Judge()

    # raise error if judge_settings is not a valid dictionary
    with pytest.raises(
        TypeError,
        match="judge_settings must be a dictionary",
    ):
        Judge(
            completed_responses="completed_responses",
            judge_settings="not_a_dict",
            template_prompt="template_prompt",
        )

    # passes
    cr = [
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
            "parameters": {"temperature": 0.5},
        },
    }
    judge = Judge(
        completed_responses=cr, judge_settings=js, template_prompt="template_prompt"
    )
    assert judge.completed_responses == cr
    assert judge.judge_settings == js
    assert judge.template_prompt == "template_prompt"


def test_judge_create_judge_inputs():
    cr = [
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

    judge = Judge(completed_responses=cr, judge_settings=js, template_prompt=tp)

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

    # "judge1" case
    judge_1_inputs = judge.create_judge_inputs("judge1")
    assert len(judge_1_inputs) == 2
    assert judge_1_inputs == [
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

    # "judge2" case
    judge_2_inputs = judge.create_judge_inputs("judge2")
    assert len(judge_2_inputs) == 2
    assert judge_2_inputs == [
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

    # "judge1, judge2" case
    judge_1_2_inputs = judge.create_judge_inputs(["judge1", "judge2"])
    assert len(judge_1_2_inputs) == 4
    assert judge_1_2_inputs == [
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


def test_judge_create_judge_file(temporary_data_folder_judge):
    cr = [
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

    judge = Judge(completed_responses=cr, judge_settings=js, template_prompt=tp)

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

    # check the judge file was created
    assert os.path.isfile("judge_file.jsonl")

    # read and check the contents of the judge file
    with open("judge_file.jsonl", "r") as f:
        judge_inputs = [dict(json.loads(line)) for line in f]

    assert len(judge_inputs) == 2
    assert judge_inputs == [
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
