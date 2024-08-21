import asyncio
import json
import logging
import os
from unittest.mock import AsyncMock, patch

import pytest
import regex as re

from prompto.experiment import Experiment
from prompto.settings import Settings
from prompto.utils import create_folder

pytest_plugins = ("pytest_asyncio",)

PROMPT_DICTS_TO_TEST = [
    # successful prompt dict with id
    {
        "id": "test_id",
        "api": "test",
        "model_name": "test_model",
        "prompt": "test prompt",
        "parameters": {"raise_error": "False"},
    },
    # successful prompt dict without id
    {
        "api": "test",
        "model_name": "test_model",
        "prompt": "test prompt",
        "parameters": {"raise_error": "False"},
    },
    # error prompt dict (ValueError)
    {
        "id": "test_id-1",
        "api": "test",
        "model_name": "test_model",
        "prompt": "test prompt",
        "parameters": {"raise_error": "True"},
    },
    # error prompt dict (NotImplementedError)
    {
        "id": "test_id-2",
        "api": "api-that-does-not-exist",
        "model_name": "test_model",
        "prompt": "test prompt",
        "parameters": {"raise_error": "False"},
    },
    # error prompt dict (Exception) - this should get retried until max_attempts
    {
        "id": "test_id-3",
        "api": "test",
        "model_name": "test_model",
        "prompt": "test prompt",
        "parameters": {"raise_error": "True", "raise_error_type": "Exception"},
    },
]


@pytest.mark.asyncio
async def test_send_requests(temporary_data_folder_for_processing, caplog, capsys):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=4)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    prompt_dicts, responses = await experiment.send_requests(
        prompt_dicts=PROMPT_DICTS_TO_TEST, attempt=1, rate_limit=60
    )

    assert prompt_dicts == PROMPT_DICTS_TO_TEST
    assert len(responses) == len(PROMPT_DICTS_TO_TEST)
    assert isinstance(responses[0], dict)
    assert responses[0]["response"] == "This is a test response"
    assert isinstance(responses[1], dict)
    assert responses[1]["response"] == "This is a test response"
    assert isinstance(responses[2], dict)
    assert (
        responses[2]["response"]
        == "ValueError - This is a test error which we should handle and return"
    )
    assert isinstance(responses[3], dict)
    assert (
        responses[3]["response"]
        == "NotImplementedError - API api-that-does-not-exist not recognised or implemented"
    )
    with pytest.raises(
        Exception, match="This is a test error which we should handle and return"
    ):
        raise responses[4]

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 5 queries at 60 QPM with RI of 1.0s (attempt 1/4)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 1/4)"
    assert print_msg in captured.err

    # check log messages
    log_msg = (
        "Response received for model test (i=1, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=test_id-1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=4, id=test_id-2): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=5, id=test_id-3) on attempt 1 of 4: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_send_requests_max(temporary_data_folder_for_processing, caplog, capsys):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=4)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    prompt_dicts, responses = await experiment.send_requests(
        prompt_dicts=PROMPT_DICTS_TO_TEST, attempt=4, rate_limit=100
    )

    assert prompt_dicts == PROMPT_DICTS_TO_TEST
    assert len(responses) == len(PROMPT_DICTS_TO_TEST)
    assert isinstance(responses[0], dict)
    assert responses[0]["response"] == "This is a test response"
    assert isinstance(responses[1], dict)
    assert responses[1]["response"] == "This is a test response"
    assert isinstance(responses[2], dict)
    assert (
        responses[2]["response"]
        == "ValueError - This is a test error which we should handle and return"
    )
    assert isinstance(responses[3], dict)
    assert (
        responses[3]["response"]
        == "NotImplementedError - API api-that-does-not-exist not recognised or implemented"
    )
    assert isinstance(responses[4], dict)
    assert responses[4]["response"] == (
        "An unexpected error occurred when querying the API: "
        "(Exception - This is a test error which we should handle and return) "
        "after maximum 4 attempts"
    )

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 5 queries at 100 QPM with RI of 0.6s (attempt 4/4)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 4/4)"
    assert print_msg in captured.err

    # check log messages
    log_msg = (
        "Response received for model test (i=1, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=test_id-1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=4, id=test_id-2): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=5, id=test_id-3) after maximum 4 attempts: "
        "Exception - This is a test error which we should handle and return"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_send_requests_with_group(
    temporary_data_folder_for_processing, caplog, capsys
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=3)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    prompt_dicts, responses = await experiment.send_requests(
        prompt_dicts=PROMPT_DICTS_TO_TEST, attempt=2, rate_limit=80, group="test_group"
    )

    assert prompt_dicts == PROMPT_DICTS_TO_TEST
    assert len(responses) == len(PROMPT_DICTS_TO_TEST)
    assert isinstance(responses[0], dict)
    assert responses[0]["response"] == "This is a test response"
    assert isinstance(responses[1], dict)
    assert responses[1]["response"] == "This is a test response"
    assert isinstance(responses[2], dict)
    assert (
        responses[2]["response"]
        == "ValueError - This is a test error which we should handle and return"
    )
    assert isinstance(responses[3], dict)
    assert (
        responses[3]["response"]
        == "NotImplementedError - API api-that-does-not-exist not recognised or implemented"
    )
    with pytest.raises(
        Exception, match="This is a test error which we should handle and return"
    ):
        raise responses[4]

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 5 queries at 80 QPM with RI of 0.75s for group 'test_group' (attempt 2/3)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test_group' (attempt 2/3)"
    assert print_msg in captured.err

    # check log messages
    log_msg = (
        "Response received for model test (i=1, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=test_id-1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=4, id=test_id-2): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=5, id=test_id-3) on attempt 2 of 3: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_send_requests_with_group_max(
    temporary_data_folder_for_processing, caplog, capsys
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=6)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    prompt_dicts, responses = await experiment.send_requests(
        prompt_dicts=PROMPT_DICTS_TO_TEST, attempt=6, rate_limit=150, group="test_group"
    )

    assert prompt_dicts == PROMPT_DICTS_TO_TEST
    assert len(responses) == len(PROMPT_DICTS_TO_TEST)
    assert isinstance(responses[0], dict)
    assert responses[0]["response"] == "This is a test response"
    assert isinstance(responses[1], dict)
    assert responses[1]["response"] == "This is a test response"
    assert isinstance(responses[2], dict)
    assert (
        responses[2]["response"]
        == "ValueError - This is a test error which we should handle and return"
    )
    assert isinstance(responses[3], dict)
    assert (
        responses[3]["response"]
        == "NotImplementedError - API api-that-does-not-exist not recognised or implemented"
    )
    assert isinstance(responses[4], dict)
    assert responses[4]["response"] == (
        "An unexpected error occurred when querying the API: "
        "(Exception - This is a test error which we should handle and return) "
        "after maximum 6 attempts"
    )

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 5 queries at 150 QPM with RI of 0.4s for group 'test_group' (attempt 6/6)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test_group' (attempt 6/6)"
    assert print_msg in captured.err

    # check log messages
    log_msg = (
        "Response received for model test (i=1, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=test_id-1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=4, id=test_id-2): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=5, id=test_id-3) after maximum 6 attempts: "
        "Exception - This is a test error which we should handle and return"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_send_requests_retry(
    temporary_data_folder_for_processing, caplog, capsys
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    await experiment.send_requests_retry(
        prompt_dicts=PROMPT_DICTS_TO_TEST, rate_limit=120
    )

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == len(PROMPT_DICTS_TO_TEST)

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 5 queries at 120 QPM with RI of 0.5s (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Sending 1 queries at 120 QPM with RI of 0.5s (attempt 2/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 2/2)"
    assert print_msg in captured.err

    # check log messages
    log_msg = (
        "Response received for model test (i=1, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=test_id-1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=4, id=test_id-2): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=5, id=test_id-3) on attempt 1 of 2: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text
    log_msg = "Retrying 1 failed queries - attempt 2 of 2..."
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=1, id=test_id-3) after maximum 2 attempts: "
        "Exception - This is a test error which we should handle and return"
    )
    assert log_msg in caplog.text
    log_msg = "Maximum attempts reached. Exiting..."
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_send_requests_retry_no_retries(
    temporary_data_folder_for_processing, caplog, capsys
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    await experiment.send_requests_retry(
        prompt_dicts=PROMPT_DICTS_TO_TEST[:-1], rate_limit=120
    )

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == len(PROMPT_DICTS_TO_TEST) - 1

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 4 queries at 120 QPM with RI of 0.5s (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 1/2)"
    assert print_msg in captured.err

    # check log messages
    log_msg = (
        "Response received for model test (i=1, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=test_id-1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=4, id=test_id-2): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text
    log_msg = "No remaining failed queries!"
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_send_requests_retry_with_group(
    temporary_data_folder_for_processing, caplog, capsys
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=3)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    await experiment.send_requests_retry(
        prompt_dicts=PROMPT_DICTS_TO_TEST, rate_limit=240, group="test_group"
    )

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == len(PROMPT_DICTS_TO_TEST)

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 5 queries at 240 QPM with RI of 0.25s for group 'test_group' (attempt 1/3)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test_group' (attempt 1/3)"
    assert print_msg in captured.err
    print_msg = "Sending 1 queries at 240 QPM with RI of 0.25s for group 'test_group' (attempt 2/3)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test_group' (attempt 2/3)"
    assert print_msg in captured.err
    print_msg = "Sending 1 queries at 240 QPM with RI of 0.25s for group 'test_group' (attempt 3/3)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test_group' (attempt 3/3)"
    assert print_msg in captured.err

    # check log messages
    log_msg = (
        "Response received for model test (i=1, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=test_id-1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=4, id=test_id-2): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=5, id=test_id-3) on attempt 1 of 3: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text
    log_msg = "Retrying 1 failed queries - attempt 2 of 3..."
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=1, id=test_id-3) on attempt 2 of 3: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text
    log_msg = "Retrying 1 failed queries - attempt 3 of 3..."
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=1, id=test_id-3) after maximum 3 attempts: "
        "Exception - This is a test error which we should handle and return"
    )
    assert log_msg in caplog.text
    log_msg = "Maximum attempts reached for group 'test_group'. Exiting..."
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_send_requests_retry_no_retries_group(
    temporary_data_folder_for_processing, caplog, capsys
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    await experiment.send_requests_retry(
        prompt_dicts=PROMPT_DICTS_TO_TEST[:-1], rate_limit=120, group="test_group"
    )

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == len(PROMPT_DICTS_TO_TEST) - 1

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 4 queries at 120 QPM with RI of 0.5s for group 'test_group' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test_group' (attempt 1/2)"
    assert print_msg in captured.err

    # check log messages
    log_msg = (
        "Response received for model test (i=1, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=test_id-1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=4, id=test_id-2): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text
    log_msg = "No remaining failed queries for group 'test_group'!"
    assert log_msg in caplog.text
