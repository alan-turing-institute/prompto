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


@pytest.mark.asyncio
async def test_query_model_and_record_response(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # query model and record response
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    assert result["id"] == "test_id"
    assert result["api"] == "test"
    assert result["model_name"] == "test_model"
    assert result["prompt"] == "test prompt"
    assert result["parameters"] == {"raise_error": "False"}
    assert result["response"] == "This is a test response"
    assert "timestamp_sent" in result.keys()

    # check logs for success message
    log_msg = (
        "Response received for model test (i=2, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 1
    assert responses[0]["id"] == result["id"]
    assert responses[0]["api"] == result["api"]
    assert responses[0]["model_name"] == result["model_name"]
    assert responses[0]["prompt"] == result["prompt"]
    assert responses[0]["parameters"] == result["parameters"]
    assert responses[0]["response"] == result["response"]
    assert "timestamp_sent" in responses[0].keys()


@pytest.mark.asyncio
async def test_query_model_and_record_response_no_index_without_id(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # query model and record response
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=None,
        attempt=1,
    )

    # check logs for success message
    log_msg = (
        "Response received for model test (i=NA, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_query_model_and_record_response_max_attepts_error(
    temporary_data_folder_for_processing,
):
    settings = Settings(data_folder="data", max_attempts=2)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # check raises error if max_attempts is reached
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Attempt number (4) cannot be greater than settings.max_attempts (2)"
        ),
    ):
        await experiment.query_model_and_record_response(
            prompt_dict={
                "id": "test_id",
                "api": "test",
                "model_name": "test_model",
                "prompt": "test prompt",
                "parameters": {"raise_error": "True"},
            },
            index=2,
            attempt=4,
        )


@pytest.mark.asyncio
async def test_query_model_and_record_response_not_implemented_error(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # we do not retry on NotImplementedError
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "api-that-does-not-exist",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    assert result["id"] == "test_id"
    assert result["api"] == "api-that-does-not-exist"
    assert result["model_name"] == "test_model"
    assert result["prompt"] == "test prompt"
    assert result["parameters"] == {"raise_error": "False"}
    assert (
        result["response"]
        == "NotImplementedError - API api-that-does-not-exist not recognised or implemented"
    )

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 1
    assert responses[0]["id"] == result["id"]
    assert responses[0]["api"] == result["api"]
    assert responses[0]["model_name"] == result["model_name"]
    assert responses[0]["prompt"] == result["prompt"]
    assert responses[0]["parameters"] == result["parameters"]
    assert responses[0]["response"] == result["response"]

    # check logs
    log_msg = (
        "Error (i=2, id=test_id): "
        "NotImplementedError - API api-that-does-not-exist "
        "not recognised or implemented"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
@patch("prompto.experiment.Experiment.generate_text", new_callable=AsyncMock)
async def test_query_model_and_record_response_key_error(
    mock_generate_text, temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # mock the generate_text method to raise a KeyError
    mock_generate_text.side_effect = KeyError("some key error")

    # we do not retry on KeyError
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    assert result["id"] == "test_id"
    assert result["api"] == "test"
    assert result["model_name"] == "test_model"
    assert result["prompt"] == "test prompt"
    assert result["parameters"] == {"raise_error": "False"}
    assert result["response"] == "KeyError - 'some key error'"

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 1
    assert responses[0]["id"] == result["id"]
    assert responses[0]["api"] == result["api"]
    assert responses[0]["model_name"] == result["model_name"]
    assert responses[0]["prompt"] == result["prompt"]
    assert responses[0]["parameters"] == result["parameters"]
    assert responses[0]["response"] == result["response"]

    # check logs
    log_msg = "Error (i=2, id=test_id): KeyError - 'some key error'"
    assert log_msg in caplog.text


@pytest.mark.asyncio
@patch("prompto.experiment.Experiment.generate_text", new_callable=AsyncMock)
async def test_query_model_and_record_response_value_error(
    mock_generate_text, temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # mock the generate_text method to raise a NotImplementedError
    mock_generate_text.side_effect = ValueError("some value error")

    # we do not retry on NotImplementedError
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    assert result["id"] == "test_id"
    assert result["api"] == "test"
    assert result["model_name"] == "test_model"
    assert result["prompt"] == "test prompt"
    assert result["parameters"] == {"raise_error": "False"}
    assert result["response"] == "ValueError - some value error"

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 1
    assert responses[0]["id"] == result["id"]
    assert responses[0]["api"] == result["api"]
    assert responses[0]["model_name"] == result["model_name"]
    assert responses[0]["prompt"] == result["prompt"]
    assert responses[0]["parameters"] == result["parameters"]
    assert responses[0]["response"] == result["response"]

    # check logs
    log_msg = "Error (i=2, id=test_id): ValueError - some value error"
    assert log_msg in caplog.text


@pytest.mark.asyncio
@patch("prompto.experiment.Experiment.generate_text", new_callable=AsyncMock)
async def test_query_model_and_record_response_type_error(
    mock_generate_text, temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # mock the generate_text method to raise a TypeError
    mock_generate_text.side_effect = TypeError("some type error")

    # we do not retry on TypeError
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    assert result["id"] == "test_id"
    assert result["api"] == "test"
    assert result["model_name"] == "test_model"
    assert result["prompt"] == "test prompt"
    assert result["parameters"] == {"raise_error": "False"}
    assert result["response"] == "TypeError - some type error"

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 1
    assert responses[0]["id"] == result["id"]
    assert responses[0]["api"] == result["api"]
    assert responses[0]["model_name"] == result["model_name"]
    assert responses[0]["prompt"] == result["prompt"]
    assert responses[0]["parameters"] == result["parameters"]
    assert responses[0]["response"] == result["response"]

    # check logs
    log_msg = "Error (i=2, id=test_id): TypeError - some type error"
    assert log_msg in caplog.text


@pytest.mark.asyncio
@patch("prompto.experiment.Experiment.generate_text", new_callable=AsyncMock)
async def test_query_model_and_record_response_file_not_found_error(
    mock_generate_text, temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # mock the generate_text method to raise a FileNotFoundError
    mock_generate_text.side_effect = FileNotFoundError("some type error")

    # we do not retry on FileNotFoundError
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    assert result["id"] == "test_id"
    assert result["api"] == "test"
    assert result["model_name"] == "test_model"
    assert result["prompt"] == "test prompt"
    assert result["parameters"] == {"raise_error": "False"}
    assert result["response"] == "FileNotFoundError - some type error"

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 1
    assert responses[0]["id"] == result["id"]
    assert responses[0]["api"] == result["api"]
    assert responses[0]["model_name"] == result["model_name"]
    assert responses[0]["prompt"] == result["prompt"]
    assert responses[0]["parameters"] == result["parameters"]
    assert responses[0]["response"] == result["response"]

    # check logs
    log_msg = "Error (i=2, id=test_id): FileNotFoundError - some type error"
    assert log_msg in caplog.text


@pytest.mark.asyncio
@patch("prompto.experiment.Experiment.generate_text", new_callable=AsyncMock)
async def test_query_model_and_record_response_exception_error(
    mock_generate_text, temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # mock the generate_text method to raise a NotImplementedError
    mock_generate_text.side_effect = Exception("some exception error")

    # we retry on Exception if attempt < max_attempts, so we should just return the error
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    with pytest.raises(Exception, match="some exception error"):
        raise result

    # check logs
    log_msg = (
        "Error (i=2, id=test_id) on attempt 1 of 2: "
        "Exception - some exception error. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
@patch("prompto.experiment.Experiment.generate_text", new_callable=AsyncMock)
async def test_query_model_and_record_response_exception_error_max(
    mock_generate_text, temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # mock the generate_text method to raise a NotImplementedError
    mock_generate_text.side_effect = Exception("some exception error")

    # no retry on Exception if attempt == max_attempts,
    # so we should just save the error to the response
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=2,
    )

    assert result["id"] == "test_id"
    assert result["api"] == "test"
    assert result["model_name"] == "test_model"
    assert result["prompt"] == "test prompt"
    assert result["parameters"] == {"raise_error": "False"}
    assert result["response"] == (
        "An unexpected error occurred when querying the API: "
        "(Exception - some exception error) "
        "after maximum 2 attempts"
    )

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 1
    assert responses[0]["id"] == result["id"]
    assert responses[0]["api"] == result["api"]
    assert responses[0]["model_name"] == result["model_name"]
    assert responses[0]["prompt"] == result["prompt"]
    assert responses[0]["parameters"] == result["parameters"]
    assert responses[0]["response"] == result["response"]

    # check logs
    log_msg = (
        "Error (i=2, id=test_id) after maximum 2 attempts: "
        "Exception - some exception error"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
@patch("prompto.experiment.Experiment.generate_text", new_callable=AsyncMock)
async def test_query_model_and_record_response_cancelled_error(
    mock_generate_text, temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # mock the generate_text method to raise a NotImplementedError
    mock_generate_text.side_effect = asyncio.CancelledError(
        "some asyncio.CancelledError error"
    )

    # we retry on asyncio.CancelledError if attempt < max_attempts, so we should just return the error
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    with pytest.raises(
        Exception, match="CancelledError - some asyncio.CancelledError error"
    ):
        raise result

    # check logs
    log_msg = (
        "Error (i=2, id=test_id) on attempt 1 of 2: "
        "CancelledError - some asyncio.CancelledError error. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
@patch("prompto.experiment.Experiment.generate_text", new_callable=AsyncMock)
async def test_query_model_and_record_response_timeout_error(
    mock_generate_text, temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2)
    experiment = Experiment("test_experiment.jsonl", settings=settings)
    create_folder(experiment.output_folder)

    # mock the generate_text method to raise a NotImplementedError
    mock_generate_text.side_effect = asyncio.TimeoutError(
        "some asyncio.TimeoutError error"
    )

    # we retry on asyncio.TimeoutError if attempt < max_attempts, so we should just return the error
    result = await experiment.query_model_and_record_response(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "model_name": "test_model",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
        attempt=1,
    )

    with pytest.raises(
        Exception, match="TimeoutError - some asyncio.TimeoutError error"
    ):
        raise result

    # check logs
    log_msg = (
        "Error (i=2, id=test_id) on attempt 1 of 2: "
        "TimeoutError - some asyncio.TimeoutError error. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text
