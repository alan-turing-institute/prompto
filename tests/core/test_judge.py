import logging

import pytest

from prompto.judge import Judge, parse_judge_arg, parse_judge_location_arg


def test_parse_judge_arg():
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
    caplog.set_level(logging.INFO)

    assert parse_judge_arg("judge1, judge2") == ["judge1", "judge2"]
    assert "Judges to be used: ['judge1', 'judge2']" in caplog.text

    assert parse_judge_arg("judge_3,judge_4,judge5") == ["judge_3", "judge_4", "judge5"]
    assert "Judges to be used: ['judge_3', 'judge_4', 'judge5']" in caplog.text


def test_parse_judge_location_arg(temporary_data_folder_judge):
    template_prompt, judge_settings = parse_judge_location_arg("judge_loc")
    assert template_prompt == (
        "This is a template prompt where you have an input "
        "{INPUT_PROMPT} and {OUTPUT_RESPONSE}"
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
        match="Settings file 'judge_loc_no_settings/settings.json' does not exist",
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
    assert Judge.check_judge_in_judge_settings("judge1", {"judge1": {}, "judge2": {}})
    assert Judge.check_judge_in_judge_settings(["judge1"], {"judge1": {}, "judge2": {}})
    assert Judge.check_judge_in_judge_settings(
        ["judge1", "judge2"], {"judge1": {}, "judge2": {}}
    )
