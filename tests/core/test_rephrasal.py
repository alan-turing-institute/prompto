import json
import logging
import os

import pytest

from prompto.rephrasal import Rephraser, load_rephrase_folder

INPUT_PROMPTS = [
    {"id": 0, "prompt": "test prompt 1", "api": "some_api", "model_name": "some_model"},
    {"id": 1, "prompt": "test prompt 2", "api": "some_api", "model_name": "some_model"},
]
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


def test_load_rephrase_folder(temporary_data_folder_rephrase):
    # test the function reads template.txt and settings.json correctly
    template_prompt, rephrase_settings = load_rephrase_folder("rephrase_loc")
    assert template_prompt == [
        "Template 1: {INPUT_PROMPT}",
        "Template 2: \n{INPUT_PROMPT}",
    ]
    assert rephrase_settings == {
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


def test_load_rephrase_folder_string_as_arg(temporary_data_folder_rephrase):
    # test the function reads template.txt and settings.json correctly
    template_prompt, rephrase_settings = load_rephrase_folder(
        "rephrase_loc", templates="template.txt"
    )
    assert template_prompt == [
        "Template 1: {INPUT_PROMPT}",
        "Template 2: \n{INPUT_PROMPT}",
    ]
    assert rephrase_settings == {
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


def test_load_rephrase_folder_arg_error(temporary_data_folder_rephrase):
    # raise error if rephrase folder is not a valid path to a directory
    with pytest.raises(
        ValueError,
        match="rephrase folder 'non_existent_folder' must be a valid path to a folder",
    ):
        load_rephrase_folder("non_existent_folder")

    # raise error if template file does not exist in the rephrase folder
    # default template.txt case
    with pytest.raises(
        FileNotFoundError,
        match="Template file 'rephrase_loc_no_template/template.txt' does not exist",
    ):
        load_rephrase_folder("rephrase_loc_no_template")

    # string template case
    with pytest.raises(
        FileNotFoundError,
        match="Template file 'rephrase_loc/some-other-template.txt' does not exist",
    ):
        load_rephrase_folder("rephrase_loc", templates="some-other-template.txt")

    # raise error if template file is not a .txt file
    with pytest.raises(
        ValueError,
        match="Template file 'rephrase_loc/template.json' must end with '.txt'",
    ):
        load_rephrase_folder("rephrase_loc", templates="template.json")

    # raise error if settings file does not exist in the rephrase folder
    with pytest.raises(
        FileNotFoundError,
        match="Rephraser settings file 'rephrase_loc_no_settings/settings.json' does not exist",
    ):
        load_rephrase_folder("rephrase_loc_no_settings")


def test_rephrase_check_rephrase_settings():
    # raise error if nothing is provided
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'rephrase_settings'"
    ):
        Rephraser.check_rephrase_settings()

    # raise error if rephrase settings is not a dictionary
    with pytest.raises(
        TypeError,
        match="rephrase_settings must be a dictionary",
    ):
        Rephraser.check_rephrase_settings("not_a_dict")

    # raise error if a value is not a dictionary
    with pytest.raises(
        TypeError,
        match="Value for rephrase key 'rephrase1' must be a dictionary",
    ):
        Rephraser.check_rephrase_settings({"rephrase1": "not_a_dict"})

    # raise error if a rephrase settings has no "api" key
    with pytest.raises(
        KeyError,
        match="'api' key not found in settings for rephrase model 'rephrase'",
    ):
        Rephraser.check_rephrase_settings(
            {"rephrase": {"model_name": "model1", "parameters": {"temperature": 0.5}}}
        )

    # raise error if rephrase settings has no "model_name" key
    with pytest.raises(
        KeyError,
        match="'model_name' key not found in settings for rephrase model 'rephrase'",
    ):
        Rephraser.check_rephrase_settings(
            {"rephrase": {"api": "test", "parameters": {"temperature": 0.5}}}
        )

    # raise error if rephrase settings has no "parameters" key
    with pytest.raises(
        KeyError,
        match="'parameters' key not found in settings for rephrase model 'rephrase'",
    ):
        Rephraser.check_rephrase_settings(
            {"rephrase": {"api": "test", "model_name": "model1"}}
        )

    # raise error if parameters is not a dictionary
    with pytest.raises(
        TypeError,
        match="Value for 'parameters' key must be a dictionary for rephrase model 'rephrase'",
    ):
        Rephraser.check_rephrase_settings(
            {
                "rephrase": {
                    "api": "test",
                    "model_name": "model1",
                    "parameters": "not_a_dict",
                }
            }
        )

    # passes
    assert Rephraser.check_rephrase_settings(
        {
            "rephrase": {
                "api": "test",
                "model_name": "model1",
                "parameters": {"temperature": 0.5},
            }
        }
    )
    assert Rephraser.check_rephrase_settings(
        {
            "rephrase1": {
                "api": "test",
                "model_name": "model1",
                "parameters": {"temperature": 0.5},
            },
            "rephrase2": {
                "api": "test",
                "model_name": "model2",
                "parameters": {"temperature": 0.5},
            },
        }
    )


def test_check_rephrase_model_in_rephrase_settings():
    # raise error if rephrase_model is not a key in rephrase settings
    # rephrase is a string case
    with pytest.raises(
        KeyError,
        match="Rephraser 'rephrase' is not a key in rephrase_settings",
    ):
        Rephraser.check_rephrase_model_in_rephrase_settings(
            "rephrase", {"rephrase1": {}, "rephrase2": {}}
        )

    # rephrase is a list of strings case (of one string)
    with pytest.raises(
        KeyError,
        match="Rephraser 'rephrase' is not a key in rephrase_settings",
    ):
        Rephraser.check_rephrase_model_in_rephrase_settings(
            ["rephrase"], {"rephrase1": {}, "rephrase2": {}}
        )

    # rephrase is a list of strings case (of multiple strings)
    with pytest.raises(
        KeyError,
        match="Rephraser 'rephrase' is not a key in rephrase_settings",
    ):
        Rephraser.check_rephrase_model_in_rephrase_settings(
            ["rephrase1", "rephrase"], {"rephrase1": {}, "rephrase2": {}}
        )

    # rephrase is a list but some are not strings
    with pytest.raises(
        TypeError,
        match="If rephrase_model is a list, each element must be a string",
    ):
        Rephraser.check_rephrase_model_in_rephrase_settings(
            ["rephrase1", 2], {"rephrase1": {}, "rephrase2": {}}
        )

    # passes
    assert Rephraser.check_rephrase_model_in_rephrase_settings(
        rephrase_model="rephrase1", rephrase_settings={"rephrase1": {}, "rephrase2": {}}
    )
    assert Rephraser.check_rephrase_model_in_rephrase_settings(
        rephrase_model=["rephrase1"],
        rephrase_settings={"rephrase1": {}, "rephrase2": {}},
    )
    assert Rephraser.check_rephrase_model_in_rephrase_settings(
        rephrase_model=["rephrase1", "rephrase2"],
        rephrase_settings={"rephrase1": {}, "rephrase2": {}},
    )


def test_check_rephrase_init():
    # raise error if nothing is provided
    with pytest.raises(
        TypeError,
        match="missing 3 required positional arguments",
    ):
        Rephraser()

    # raise error if rephrase_settings is not a valid dictionary
    with pytest.raises(
        TypeError,
        match="rephrase_settings must be a dictionary",
    ):
        Rephraser(
            input_prompts="input_prompts (no check on list of dicts)",
            template_prompts={"template": "some template"},
            rephrase_settings="not_a_dict",
        )

    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase = Rephraser(
        input_prompts=INPUT_PROMPTS,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
    )
    assert rephrase.input_prompts == INPUT_PROMPTS
    assert rephrase.rephrase_settings == REPHRASE_SETTINGS
    assert rephrase.template_prompts == tp
    assert rephrase.rephrase_prompts == []


def test_rephrase_create_rephrase_inputs_errors():
    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase = Rephraser(
        input_prompts=INPUT_PROMPTS,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
    )

    # raise error if rephrase not provided
    with pytest.raises(
        TypeError,
        match="missing 1 required positional argument: 'rephrase_model'",
    ):
        rephrase.create_rephrase_inputs()

    # raise error if rephrase_model is not in rephrase settings
    with pytest.raises(
        KeyError,
        match="Rephraser 'rephrase' is not a key in rephrase_settings",
    ):
        rephrase.create_rephrase_inputs("rephrase")

    # raise error if rephrase_model is not in rephrase settings (list case)
    with pytest.raises(
        KeyError,
        match="Rephraser 'rephrase' is not a key in rephrase_settings",
    ):
        rephrase.create_rephrase_inputs(["rephrase", "rephrase1"])

    # raise error if rephrase_model is a list but some are not strings
    with pytest.raises(
        TypeError,
        match="If rephrase_model is a list, each element must be a string",
    ):
        rephrase.create_rephrase_inputs(["rephrase1", 2])


def test_rephrase_create_rephrase_inputs(capsys):
    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase = Rephraser(
        input_prompts=INPUT_PROMPTS,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
    )

    # "rephrase1" case
    rephrase_1_inputs = rephrase.create_rephrase_inputs("rephrase1")
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
        },
    ]
    assert len(rephrase_1_inputs) == 4
    assert rephrase_1_inputs == expected_result
    assert rephrase.rephrase_prompts == expected_result

    captured = capsys.readouterr()
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase1' and template '0'"
        in captured.err
    )
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase1' and template '1'"
        in captured.err
    )

    # "rephrase2" case
    rephrase_2_inputs = rephrase.create_rephrase_inputs("rephrase2")
    expected_result_2 = [
        {
            "id": "rephrase-rephrase2-0-0",
            "template_index": 0,
            "prompt": "Template 1: test prompt 1",
            "api": "test",
            "model_name": "model2",
            "parameters": {"temperature": 0.2, "top_k": 0.9},
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
        },
    ]
    assert len(rephrase_2_inputs) == 4
    assert rephrase_2_inputs == expected_result_2
    assert rephrase.rephrase_prompts == expected_result_2

    captured = capsys.readouterr()
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase2' and template '0'"
        in captured.err
    )
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase2' and template '1'"
        in captured.err
    )


def test_rephrase_create_rephrase_inputs_multiple_rephrase_models(capsys):
    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase = Rephraser(
        input_prompts=INPUT_PROMPTS,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
    )

    # "rephrase1, rephrase2" case
    rephrase_1_2_inputs = rephrase.create_rephrase_inputs(["rephrase1", "rephrase2"])
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
        },
    ]
    assert len(rephrase_1_2_inputs) == 8
    assert rephrase_1_2_inputs == expected_result
    assert rephrase.rephrase_prompts == expected_result

    captured = capsys.readouterr()
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase1' and template '0'"
        in captured.err
    )
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase1' and template '1'"
        in captured.err
    )
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase2' and template '0'"
        in captured.err
    )
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase2' and template '1'"
        in captured.err
    )


def test_rephrase_create_rephrase_file(temporary_data_folder_rephrase, capsys):
    # case where template_prompt has multiple templates
    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase = Rephraser(
        input_prompts=INPUT_PROMPTS,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
    )

    # raise error if nothing is provided
    with pytest.raises(
        TypeError,
        match="missing 2 required positional arguments",
    ):
        rephrase.create_rephrase_file()

    # raise error if out_filepath is not a string that ends with ".jsonl"
    with pytest.raises(
        ValueError,
        match="out_filepath must end with '.jsonl'",
    ):
        rephrase.create_rephrase_file(
            rephrase_model="rephrase", out_filepath="rephrase_file"
        )

    # create rephrase file
    rephrase.create_rephrase_file(
        rephrase_model="rephrase1", out_filepath="rephrase_file.jsonl"
    )

    captured = capsys.readouterr()
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase1' and template '0'"
        in captured.err
    )
    assert (
        "Creating rephrase inputs for rephrase model 'rephrase1' and template '1'"
        in captured.err
    )

    # check the rephrase file was created
    assert os.path.isfile("rephrase_file.jsonl")

    # read and check the contents of the rephrase file
    with open("rephrase_file.jsonl", "r") as f:
        rephrase_inputs = [dict(json.loads(line)) for line in f]

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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
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
            "input-api": "some_api",
            "input-model_name": "some_model",
        },
    ]

    assert len(rephrase_inputs) == 4
    assert rephrase_inputs == expected_result
    assert rephrase.rephrase_prompts == expected_result


def test_rephrase_convert_rephrased_prompt_dict_to_input():
    rephrased_prompt = {
        "id": "rephrase-rephrase1-0-0",
        "template_index": 0,
        "prompt": "Template 1: test prompt 1",
        "response": "Rephrased test prompt 1.0",
        "api": "test",
        "model_name": "model1",
        "parameters": {"temperature": 0.5},
        "input-id": 0,
        "input-prompt": "test prompt 1",
        "input-api": "some_api",
        "input-model_name": "some_model",
        "input-parameters": {"temperature": 0.9},
    }

    assert Rephraser._convert_rephrased_prompt_dict_to_input(rephrased_prompt) == {
        "id": "rephrase-rephrase1-0-0",
        "prompt": "Rephrased test prompt 1.0",
        "input-prompt": "test prompt 1",
        "input-id": 0,
        "api": rephrased_prompt["input-api"],
        "model_name": rephrased_prompt["input-model_name"],
        "parameters": rephrased_prompt["input-parameters"],
    }


def test_rephrase_create_new_input_file_keep_original(
    temporary_data_folder_rephrase, caplog, capsys
):
    caplog.set_level(logging.INFO)

    # case where template_prompt has multiple templates
    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase = Rephraser(
        input_prompts=INPUT_PROMPTS,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
    )

    rephrased_prompts = rephrase.create_rephrase_inputs("rephrase1")
    for i in range(len(rephrased_prompts)):
        rephrased_prompts[i] = {
            **rephrased_prompts[i],
            "response": f"Rephrased test prompt {i}.0",
        }

    # raise error if out_filepath is not a string that ends with ".jsonl"
    with pytest.raises(
        ValueError,
        match="out_filepath must end with '.jsonl'",
    ):
        rephrase.create_new_input_file(
            keep_original=True,
            completed_rephrase_responses=rephrased_prompts,
            out_filepath="new_input_file",
        )

    rephrase.create_new_input_file(
        keep_original=True,
        completed_rephrase_responses=rephrased_prompts,
        out_filepath="new_input_file.jsonl",
    )

    assert (
        "Creating new input file with rephrased prompts at new_input_file.jsonl..."
        in caplog.text
    )
    captured = capsys.readouterr()
    assert "Writing new input prompts to new_input_file.jsonl" in captured.err

    # check the new input file was created
    assert os.path.isfile("new_input_file.jsonl")

    # read and check the contents of the new input file
    with open("new_input_file.jsonl", "r") as f:
        new_inputs = [dict(json.loads(line)) for line in f]

    expected_result = [
        {
            "id": "rephrase-rephrase1-0-0",
            "prompt": rephrased_prompts[0]["response"],
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase1-0-1",
            "prompt": rephrased_prompts[1]["response"],
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase1-1-0",
            "prompt": rephrased_prompts[2]["response"],
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase1-1-1",
            "prompt": rephrased_prompts[3]["response"],
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "api": "some_api",
            "model_name": "some_model",
        },
        INPUT_PROMPTS[0] | {"input-id": INPUT_PROMPTS[0]["id"]},
        INPUT_PROMPTS[1] | {"input-id": INPUT_PROMPTS[1]["id"]},
    ]

    assert len(new_inputs) == 6
    assert new_inputs == expected_result


def test_rephrase_create_new_input_file_remove_original(
    temporary_data_folder_rephrase, caplog, capsys
):
    # test case where the original prompts are not kept (only rephrased prompts are taken)
    caplog.set_level(logging.INFO)

    # case where template_prompt has multiple templates
    tp = ["Template 1: {INPUT_PROMPT}", "Template 2: \n{INPUT_PROMPT}"]
    rephrase = Rephraser(
        input_prompts=INPUT_PROMPTS,
        template_prompts=tp,
        rephrase_settings=REPHRASE_SETTINGS,
    )

    rephrased_prompts = rephrase.create_rephrase_inputs(["rephrase1", "rephrase2"])
    for i in range(len(rephrased_prompts)):
        rephrased_prompts[i] = {
            **rephrased_prompts[i],
            "response": f"Rephrased test prompt {i}.0",
        }

    # raise error if out_filepath is not a string that ends with ".jsonl"
    with pytest.raises(
        ValueError,
        match="out_filepath must end with '.jsonl'",
    ):
        rephrase.create_new_input_file(
            keep_original=False,
            completed_rephrase_responses=rephrased_prompts,
            out_filepath="new_input_file",
        )

    rephrase.create_new_input_file(
        keep_original=False,
        completed_rephrase_responses=rephrased_prompts,
        out_filepath="new_input_file.jsonl",
    )

    assert (
        "Creating new input file with rephrased prompts at new_input_file.jsonl..."
        in caplog.text
    )
    captured = capsys.readouterr()
    assert "Writing new input prompts to new_input_file.jsonl" in captured.err

    # check the new input file was created
    assert os.path.isfile("new_input_file.jsonl")

    # read and check the contents of the new input file
    with open("new_input_file.jsonl", "r") as f:
        new_inputs = [dict(json.loads(line)) for line in f]

    expected_result = [
        {
            "id": "rephrase-rephrase1-0-0",
            "prompt": rephrased_prompts[0]["response"],
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase1-0-1",
            "prompt": rephrased_prompts[1]["response"],
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase1-1-0",
            "prompt": rephrased_prompts[2]["response"],
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase1-1-1",
            "prompt": rephrased_prompts[3]["response"],
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase2-0-0",
            "prompt": rephrased_prompts[4]["response"],
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase2-0-1",
            "prompt": rephrased_prompts[5]["response"],
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase2-1-0",
            "prompt": rephrased_prompts[6]["response"],
            "input-id": 0,
            "input-prompt": "test prompt 1",
            "api": "some_api",
            "model_name": "some_model",
        },
        {
            "id": "rephrase-rephrase2-1-1",
            "prompt": rephrased_prompts[7]["response"],
            "input-id": 1,
            "input-prompt": "test prompt 2",
            "api": "some_api",
            "model_name": "some_model",
        },
    ]

    assert len(new_inputs) == 8
    assert new_inputs == expected_result
