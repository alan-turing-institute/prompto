import logging
import os

import pytest

from batch_llm.utils import (
    create_folder,
    log_error_response_chat,
    log_error_response_query,
    log_success_response_chat,
    log_success_response_query,
    move_file,
    sort_jsonl_files_by_creation_time,
    write_log_message,
)


def test_sort_jsonl_files_by_creation_time(temporary_data_folders):
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
        "Response received for model test (i=0)\n"
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
        "Response received for model test (i=0, message=3/4)\n"
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

    expected_log_message = (
        "Error with model test (i=0)\n" "Prompt: test prompt...\n" "Error: test error"
    )
    assert log_message == expected_log_message
    assert expected_log_message in caplog.text


def test_log_error_response_chat(caplog):
    caplog.set_level(logging.INFO)

    # raise error if no index, model, message_index, n_messages, message, responses_so_far, error_as_string is passed
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
        "Error with model test (i=0, message=3/4)\n"
        "Prompt: test prompt...\n"
        "Responses so far: ['hi', 'hello']...\n"
        "Error: test error"
    )
    assert log_message == expected_log_message
    assert expected_log_message in caplog.text
