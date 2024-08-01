import logging
import os

import pytest
import regex as re

from prompto.utils import (
    check_either_required_env_variables_set,
    check_max_queries_dict,
    check_optional_env_variables_set,
    check_required_env_variables_set,
    create_folder,
    get_environment_variable,
    get_model_name_identifier,
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    move_file,
    sort_jsonl_files_by_creation_time,
    sort_prompts_by_model_for_api,
    write_log_message,
)


def test_sort_jsonl_files_by_creation_time(temporary_data_folders, caplog):
    caplog.set_level(logging.INFO)
    # raise error if no input folder is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        sort_jsonl_files_by_creation_time()

    # raise error if not a path
    with pytest.raises(
        ValueError, match="Input folder 'not_a_folder' must be a valid path to a folder"
    ):
        sort_jsonl_files_by_creation_time(input_folder="not_a_folder")

    # raise error if not a folder
    with pytest.raises(
        ValueError, match="Input folder 'test.txt' must be a valid path to a folder"
    ):
        sort_jsonl_files_by_creation_time(input_folder="test.txt")

    # sort the jsonl files in the utils folder by creation time
    logging.info(
        {
            os.path.join("utils", f): {
                "ctime": os.path.getctime(os.path.join("utils", f)),
                "mtime": os.path.getmtime(os.path.join("utils", f)),
            }
            for f in os.listdir("utils")
            if f.endswith(".jsonl")
        }
    )
    sorted_files = sort_jsonl_files_by_creation_time(input_folder="utils")
    assert sorted_files == ["first.jsonl", "second.jsonl", "third.jsonl"]

    # sort empty folder should return empty list
    empty_folder = sort_jsonl_files_by_creation_time(input_folder="data")
    assert empty_folder == []


def test_create_folder(temporary_data_folders, caplog):
    caplog.set_level(logging.INFO)

    # raise error if no folder is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        create_folder()

    # create a folder
    create_folder(folder="new_folder")
    assert "new_folder" in os.listdir()
    assert "Creating folder 'new_folder'" in caplog.text

    # create a folder that already exists
    create_folder(folder="new_folder")
    assert "new_folder" in os.listdir()
    assert "Folder 'new_folder' already exists" in caplog.text


def test_move_file(temporary_data_folders, caplog):
    caplog.set_level(logging.INFO)

    # raise error if no source or destination is passed
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        move_file()

    # raise error if only source is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        move_file(source="source.txt")

    # raise error if only destination is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        move_file(destination="destination.txt")

    # raise error if source file does not exist
    with pytest.raises(FileNotFoundError, match="File 'source.txt' does not exist"):
        move_file(source="source.txt", destination="destination.txt")

    # move a file from one location to another
    with open("source.txt", "w") as f:
        f.write("source")
    move_file(source="source.txt", destination="destination.txt")

    assert "source.txt" not in os.listdir()
    assert "destination.txt" in os.listdir()
    assert "Moving file from source.txt to destination.txt" in caplog.text


def test_write_log_message(caplog):
    caplog.set_level(logging.INFO)

    # raise error if no log_file or log_message is passed
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        write_log_message()

    # raise error if only log_file is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        write_log_message(log_file="log.txt")

    # raise error if only log_message is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        write_log_message(log_message="log message")

    # write a log message to a log file
    write_log_message(log_file="log.txt", log_message="log message", log=True)

    assert "log message" in open("log.txt").read()
    assert "log message" in caplog.text

    # write a log message to a log file without logging
    write_log_message(log_file="new_log.txt", log_message="new log message", log=False)

    assert "new log message" in open("new_log.txt").read()
    assert "new log message" not in caplog.text

    # remove the log files
    os.remove("log.txt")
    os.remove("new_log.txt")


def test_log_success_response_query(caplog):
    caplog.set_level(logging.INFO)

    # raise error if no index, model, prompt, or response_text is passed
    with pytest.raises(TypeError, match="missing 4 required positional arguments"):
        log_success_response_query()

    # log a successful response from a model to a query
    log_message = log_success_response_query(
        index=0, model="test", prompt="test prompt", response_text="test response"
    )

    expected_log_message = (
        "Response received for model test (i=0, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: test response...\n"
    )
    assert log_message == expected_log_message
    assert expected_log_message in caplog.text


def test_log_success_response_chat(caplog):
    caplog.set_level(logging.INFO)

    # raise error if no index, model, message_index, n_messages, message or response_text is passed
    with pytest.raises(TypeError, match="missing 6 required positional arguments"):
        log_success_response_chat()

    # log a successful response from a model to a chat
    log_message = log_success_response_chat(
        index=0,
        model="test",
        message_index=2,
        n_messages=4,
        message="test prompt",
        response_text="test response",
    )

    expected_log_message = (
        "Response received for model test (i=0, id=NA, message=3/4)\n"
        "Prompt: test prompt...\n"
        "Response: test response...\n"
    )
    assert log_message == expected_log_message
    assert expected_log_message in caplog.text


def test_log_error_response_query(caplog):
    caplog.set_level(logging.INFO)

    # raise error if no index, model, prompt, or error_as_string is passed
    with pytest.raises(TypeError, match="missing 4 required positional arguments"):
        log_error_response_query()

    # log an error response from a model to a query
    log_message = log_error_response_query(
        index=0, model="test", prompt="test prompt", error_as_string="test error"
    )

    expected_log_message = "Error with model test (i=0, id=NA)\nPrompt: test prompt...\nError: test error\n"
    assert log_message == expected_log_message
    assert expected_log_message in caplog.text


def test_log_error_response_chat(caplog):
    caplog.set_level(logging.INFO)

    # raise error if no index, model, message_index, n_messages, message,
    # responses_so_far, error_as_string is passed
    with pytest.raises(TypeError, match="missing 7 required positional arguments"):
        log_error_response_chat()

    # log an error response from a model to a chat
    log_message = log_error_response_chat(
        index=0,
        model="test",
        message_index=2,
        n_messages=4,
        message="test prompt",
        responses_so_far=["hi", "hello"],
        error_as_string="test error",
    )

    expected_log_message = (
        "Error with model test (i=0, id=NA, message=3/4)\n"
        "Prompt: test prompt...\n"
        "Responses so far: ['hi', 'hello']...\n"
        "Error: test error\n"
    )
    assert log_message == expected_log_message
    assert expected_log_message in caplog.text


def test_check_required_env_variables_set(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "test")
    monkeypatch.setenv("TEST_VAR_2", "test")
    monkeypatch.delenv("TEST_VAR_3", raising=False)
    monkeypatch.delenv("TEST_VAR_4", raising=False)

    # check passes
    assert check_required_env_variables_set(["TEST_VAR"]) == []
    assert check_required_env_variables_set(["TEST_VAR", "TEST_VAR_2"]) == []

    # check KeyErrors are being returned within the list
    test_case = check_required_env_variables_set(["TEST_VAR_3"])
    assert len(test_case) == 1
    with pytest.raises(KeyError, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]

    test_case = check_required_env_variables_set(["TEST_VAR", "TEST_VAR_3"])
    assert len(test_case) == 1
    with pytest.raises(KeyError, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]

    test_case = check_required_env_variables_set(
        ["TEST_VAR", "TEST_VAR_2", "TEST_VAR_3"]
    )
    assert len(test_case) == 1
    with pytest.raises(KeyError, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]

    test_case = check_required_env_variables_set(["TEST_VAR_3", "TEST_VAR_4"])
    assert len(test_case) == 2
    with pytest.raises(KeyError, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]
    with pytest.raises(KeyError, match="Environment variable 'TEST_VAR_4' is not set"):
        raise test_case[1]

    test_case = check_required_env_variables_set(
        ["TEST_VAR", "TEST_VAR_2", "TEST_VAR_3", "TEST_VAR_4"]
    )
    assert len(test_case) == 2
    with pytest.raises(KeyError, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]
    with pytest.raises(KeyError, match="Environment variable 'TEST_VAR_4' is not set"):
        raise test_case[1]


def test_check_optional_env_variables_set(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "test")
    monkeypatch.setenv("TEST_VAR_2", "test")
    monkeypatch.delenv("TEST_VAR_3", raising=False)
    monkeypatch.delenv("TEST_VAR_4", raising=False)

    # check passes
    assert check_optional_env_variables_set(["TEST_VAR"]) == []
    assert check_optional_env_variables_set(["TEST_VAR", "TEST_VAR_2"]) == []

    # check Warnings are being returned within the list
    test_case = check_optional_env_variables_set(["TEST_VAR_3"])
    assert len(test_case) == 1
    with pytest.raises(Warning, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]

    test_case = check_optional_env_variables_set(["TEST_VAR", "TEST_VAR_3"])
    assert len(test_case) == 1
    with pytest.raises(Warning, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]

    test_case = check_optional_env_variables_set(
        ["TEST_VAR", "TEST_VAR_2", "TEST_VAR_3"]
    )
    assert len(test_case) == 1
    with pytest.raises(Warning, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]

    test_case = check_optional_env_variables_set(["TEST_VAR_3", "TEST_VAR_4"])
    assert len(test_case) == 2
    with pytest.raises(Warning, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]
    with pytest.raises(Warning, match="Environment variable 'TEST_VAR_4' is not set"):
        raise test_case[1]

    test_case = check_optional_env_variables_set(
        ["TEST_VAR", "TEST_VAR_2", "TEST_VAR_3", "TEST_VAR_4"]
    )
    assert len(test_case) == 2
    with pytest.raises(Warning, match="Environment variable 'TEST_VAR_3' is not set"):
        raise test_case[0]
    with pytest.raises(Warning, match="Environment variable 'TEST_VAR_4' is not set"):
        raise test_case[1]


def test_check_either_required_env_variables_set(monkeypatch):
    monkeypatch.setenv("TEST_VAR", "test")
    monkeypatch.setenv("TEST_VAR_ALT", "test")
    monkeypatch.setenv("TEST_VAR_2", "test")
    monkeypatch.delenv("TEST_VAR_2_ALT", raising=False)
    monkeypatch.delenv("TEST_VAR_3", raising=False)
    monkeypatch.delenv("TEST_VAR_3_ALT", raising=False)
    monkeypatch.delenv("TEST_VAR_4", raising=False)
    monkeypatch.delenv("TEST_VAR_4_ALT", raising=False)

    # check error raising if parameter is not a list of lists
    with pytest.raises(
        TypeError,
        match="The 'required_env_variables' parameter must be a list of lists of environment variables",
    ):
        check_either_required_env_variables_set(["TEST_VAR", "TEST_VAR_ALT"])

    # check passes
    assert check_either_required_env_variables_set([["TEST_VAR", "TEST_VAR_ALT"]]) == []

    # check Warnings and KeyErrors are being returned within the list
    test_case = check_either_required_env_variables_set(
        [["TEST_VAR_2", "TEST_VAR_2_ALT"]]
    )
    assert len(test_case) == 1
    with pytest.raises(
        Warning, match="Environment variable 'TEST_VAR_2_ALT' is not set"
    ):
        raise test_case[0]

    test_case = check_either_required_env_variables_set(
        [["TEST_VAR_3", "TEST_VAR_3_ALT"]]
    )
    assert len(test_case) == 1
    with pytest.raises(
        KeyError,
        match=re.escape(
            "At least one of the environment variables '['TEST_VAR_3', 'TEST_VAR_3_ALT']' must be set"
        ),
    ):
        raise test_case[0]

    test_case = check_either_required_env_variables_set(
        [
            ["TEST_VAR", "TEST_VAR_ALT"],
            ["TEST_VAR_2", "TEST_VAR_2_ALT"],
            ["TEST_VAR_3", "TEST_VAR_3_ALT"],
        ]
    )
    assert len(test_case) == 2
    with pytest.raises(
        Warning, match="Environment variable 'TEST_VAR_2_ALT' is not set"
    ):
        raise test_case[0]
    with pytest.raises(
        KeyError,
        match=re.escape(
            "At least one of the environment variables '['TEST_VAR_3', 'TEST_VAR_3_ALT']' must be set"
        ),
    ):
        raise test_case[1]

    test_case = check_either_required_env_variables_set(
        [
            ["TEST_VAR", "TEST_VAR_ALT"],
            ["TEST_VAR_2", "TEST_VAR_2_ALT"],
            ["TEST_VAR", "TEST_VAR_ALT", "TEST_VAR_2", "TEST_VAR_2_ALT"],
            ["TEST_VAR_3", "TEST_VAR_3_ALT"],
            ["TEST_VAR_4", "TEST_VAR_4_ALT"],
        ]
    )
    assert len(test_case) == 4
    with pytest.raises(
        Warning, match="Environment variable 'TEST_VAR_2_ALT' is not set"
    ):
        raise test_case[0]
    with pytest.raises(
        Warning, match="Environment variable 'TEST_VAR_2_ALT' is not set"
    ):
        raise test_case[1]
    with pytest.raises(
        KeyError,
        match=re.escape(
            "At least one of the environment variables '['TEST_VAR_3', 'TEST_VAR_3_ALT']' must be set"
        ),
    ):
        raise test_case[2]
    with pytest.raises(
        KeyError,
        match=re.escape(
            "At least one of the environment variables '['TEST_VAR_4', 'TEST_VAR_4_ALT']' must be set"
        ),
    ):
        raise test_case[3]


def test_get_model_name_identifier():
    # raise error if no model_name is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        get_model_name_identifier()

    assert get_model_name_identifier("test") == "test"
    assert get_model_name_identifier("test_model") == "test_model"
    assert get_model_name_identifier("test-model") == "test_model"
    assert get_model_name_identifier("test/model") == "test_model"
    assert get_model_name_identifier("test.model") == "test_model"
    assert get_model_name_identifier("test:model") == "test_model"
    assert get_model_name_identifier("test model") == "test_model"

    # test some real use cases
    assert (
        get_model_name_identifier("vicgalle/gpt2-open-instruct-v1")
        == "vicgalle_gpt2_open_instruct_v1"
    )
    assert (
        get_model_name_identifier("EleutherAI/gpt-neo-2.7B")
        == "EleutherAI_gpt_neo_2_7B"
    )


def test_sort_prompts_by_model_for_api():
    # raise error if no prompts is passed
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        sort_prompts_by_model_for_api()

    # raise error if only prompts is passed (no api passed)
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        sort_prompts_by_model_for_api(prompt_dicts=[{}])

    # raise error if only api is passed (no prompts passed)
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        sort_prompts_by_model_for_api(api="test")

    # test cases where there's only one "api" type in the list
    assert sort_prompts_by_model_for_api(prompt_dicts=[{}], api="test") == [{}]
    assert sort_prompts_by_model_for_api(
        prompt_dicts=[{"api": "test", "prompt": "test prompt"}], api="test"
    ) == [{"api": "test", "prompt": "test prompt"}]
    assert sort_prompts_by_model_for_api(
        prompt_dicts=[
            {"api": "test", "prompt": "test prompt 1"},
            {"api": "test", "model_name": "b", "prompt": "test prompt 2"},
            {"api": "test", "model_name": "a", "prompt": "test prompt 3"},
        ],
        api="test",
    ) == [
        {"api": "test", "prompt": "test prompt 1"},
        {"api": "test", "model_name": "a", "prompt": "test prompt 3"},
        {"api": "test", "model_name": "b", "prompt": "test prompt 2"},
    ]
    assert sort_prompts_by_model_for_api(
        prompt_dicts=[
            {"api": "test", "model_name": "b", "prompt": "test prompt 1"},
            {"api": "test", "model_name": "a", "prompt": "test prompt 2"},
            {"api": "test", "prompt": "test prompt 3"},
        ],
        api="test",
    ) == [
        {"api": "test", "prompt": "test prompt 3"},
        {"api": "test", "model_name": "a", "prompt": "test prompt 2"},
        {"api": "test", "model_name": "b", "prompt": "test prompt 1"},
    ]

    # test cases where there are multiple "api" types in the list
    # the items with "api" type not equal to the one passed should stay in the same locations
    assert sort_prompts_by_model_for_api(
        prompt_dicts=[
            {"api": "other", "prompt": "prompt 1"},
            {"api": "other", "prompt": "prompt 2"},
            {"api": "test", "prompt": "test prompt 1"},
            {"api": "other", "prompt": "prompt 3"},
            {"api": "test", "model_name": "b", "prompt": "test prompt 2"},
            {"api": "other", "prompt": "prompt 4"},
            {"api": "test", "model_name": "a", "prompt": "test prompt 3"},
        ],
        api="test",
    ) == [
        {"api": "other", "prompt": "prompt 1"},
        {"api": "other", "prompt": "prompt 2"},
        {"api": "test", "prompt": "test prompt 1"},
        {"api": "other", "prompt": "prompt 3"},
        {"api": "test", "model_name": "a", "prompt": "test prompt 3"},
        {"api": "other", "prompt": "prompt 4"},
        {"api": "test", "model_name": "b", "prompt": "test prompt 2"},
    ]
    assert sort_prompts_by_model_for_api(
        prompt_dicts=[
            {"api": "test", "model_name": "b", "prompt": "test prompt 1"},
            {"api": "other", "prompt": "prompt 1"},
            {"api": "test", "model_name": "a", "prompt": "test prompt 2"},
            {"api": "other", "prompt": "prompt 2"},
            {"api": "other2", "prompt": "prompt 1"},
            {"api": "test", "prompt": "test prompt 3"},
            {"api": "other", "prompt": "prompt 3"},
        ],
        api="test",
    ) == [
        {"api": "test", "prompt": "test prompt 3"},
        {"api": "other", "prompt": "prompt 1"},
        {"api": "test", "model_name": "a", "prompt": "test prompt 2"},
        {"api": "other", "prompt": "prompt 2"},
        {"api": "other2", "prompt": "prompt 1"},
        {"api": "test", "model_name": "b", "prompt": "test prompt 1"},
        {"api": "other", "prompt": "prompt 3"},
    ]


def test_get_environment_variable_main(monkeypatch):
    # raise error if no arguments are passed
    with pytest.raises(TypeError, match="missing 2 required positional argument"):
        get_environment_variable()

    # raise error if only one argument is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        get_environment_variable("env_variable")

    # raise error if only env_variable is passed
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'model_name'"
    ):
        get_environment_variable(env_variable="TEST_VAR")

    # raise error if only model_name is passed
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'env_variable'"
    ):
        get_environment_variable(model_name="TEST_MODEL_NAME")

    # get the environment variable from the main environment variable
    monkeypatch.setenv("TEST_VAR", "test")
    assert (
        get_environment_variable(env_variable="TEST_VAR", model_name="TEST_MODEL_NAME")
        == "test"
    )


def test_get_environment_variable_model_name(monkeypatch):
    # get the environment variable from the model name identifier
    monkeypatch.setenv("TEST_VAR_TEST_MODEL_NAME", "test_1")
    assert (
        get_environment_variable(env_variable="TEST_VAR", model_name="TEST_MODEL_NAME")
        == "test_1"
    )

    # get the environment variable from the model name identifier with hyphens
    monkeypatch.setenv("TEST_VAR_test_model_name", "test_2")
    assert (
        get_environment_variable(env_variable="TEST_VAR", model_name="test-model-name")
        == "test_2"
    )

    # get the environment variable from the model name identifier
    # even when the main environment variable is set
    monkeypatch.setenv("TEST_VAR", "main")
    assert (
        get_environment_variable(env_variable="TEST_VAR", model_name="TEST_MODEL_NAME")
        == "test_1"
    )


def test_get_environment_variable_error():
    # raise error if the environment variable is not set
    with pytest.raises(
        KeyError, match="Neither 'TEST_VAR' nor 'TEST_VAR_TEST_MODEL_NAME' is set"
    ):
        get_environment_variable(env_variable="TEST_VAR", model_name="TEST_MODEL_NAME")

    with pytest.raises(
        KeyError, match="Neither 'TEST_VAR' nor 'TEST_VAR_test_model_name' is set"
    ):
        get_environment_variable(env_variable="TEST_VAR", model_name="test-model-name")


def test_check_max_queries_dict():
    # raise error if no max_queries_dict is passed
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        check_max_queries_dict()

    # raise error if max_queries_dict is not a dictionary
    with pytest.raises(
        TypeError, match="max_queries_dict must be a dictionary, not <class 'str'>"
    ):
        check_max_queries_dict(max_queries_dict="test")

    # raise error if any of the keys is not a string
    with pytest.raises(
        TypeError,
        match="max_queries_dict keys must be strings, not <class 'int'>",
    ):
        check_max_queries_dict(max_queries_dict={1: 10})

    # raise error if any of the values is not an integer or dictionary
    with pytest.raises(
        TypeError,
        match="max_queries_dict values must be integers or dictionaries, not <class 'str'>",
    ):
        check_max_queries_dict(max_queries_dict={"test": "test"})

    # raise error if max_queries_dict has values which are negative integers
    with pytest.raises(
        ValueError,
        match=(
            "if a value of max_queries_dict is an integer, "
            "the value must be a positive integer, not negative"
        ),
    ):
        check_max_queries_dict(max_queries_dict={"test": -1})

    # raise error if max_queries_dict has values which are dictionaries
    # which have keys that are not strings
    with pytest.raises(
        TypeError,
        match="if a value of max_queries_dict is a dictionary, the sub-keys must be strings, not <class 'int'>",
    ):
        check_max_queries_dict(max_queries_dict={"test": {1: 10}})

    # raise error if max_queries_dict has values which are dictionaries
    # which have values that are not integers
    with pytest.raises(
        TypeError,
        match="if a value of max_queries_dict is a dictionary, the sub-values must be integers, not <class 'str'>",
    ):
        check_max_queries_dict(
            max_queries_dict={"test": {"test_sub_key": "test_sub_value"}}
        )

    # raise error if max_queries_dict has values which are dictionaries
    # which have values that are negative integers
    with pytest.raises(
        ValueError,
        match=(
            "if a value of max_queries_dict is a dictionary, "
            "the sub-values must be positive integers, not negative"
        ),
    ):
        check_max_queries_dict(max_queries_dict={"test": {"test_sub_key": -1}})

    # check passes
    assert check_max_queries_dict(max_queries_dict={"test": {}})
    assert check_max_queries_dict(max_queries_dict={"test": 1})
    assert check_max_queries_dict(max_queries_dict={"test": {"test_sub_key": 1}})
    assert check_max_queries_dict(
        max_queries_dict={"test": {"test_sub_key": 1, "test_sub_key_2": 2}, "test_2": 2}
    )
