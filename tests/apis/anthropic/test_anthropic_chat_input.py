import logging
from unittest.mock import Mock, patch

import pytest

from prompto.apis.anthropic import AnthropicAPI
from prompto.settings import Settings

from ...conftest import CopyingAsyncMock
from .test_anthropic import prompt_dict_chat

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_anthropic_query_chat_no_env_var(
    prompt_dict_chat, temporary_data_folders, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    anthropic_api = AnthropicAPI(settings=settings, log_file=log_file)

    # raise error if no environment variable is set
    with pytest.raises(
        KeyError,
        match=(
            "Neither 'ANTHROPIC_API_KEY' nor 'ANTHROPIC_API_KEY_anthropic_model_name' "
            "environment variable is set."
        ),
    ):
        await anthropic_api._query_chat(prompt_dict_chat, index=0)


@pytest.mark.asyncio
@patch("anthropic.resources.AsyncMessages.create", new_callable=CopyingAsyncMock)
@patch("prompto.apis.anthropic.anthropic.process_response", new_callable=Mock)
async def test_anthropic_query_chat(
    mock_process_response,
    mock_anthropic,
    prompt_dict_chat,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "DUMMY")
    anthropic_api = AnthropicAPI(settings=settings, log_file=log_file)

    # mock the response from the API
    # NOTE: The actual response from the API is a anthropic.types.message.Message objects
    # not a string value, but for the purpose of this test, we are using a string value
    # and testing that this is the input to the process_response function
    anthropic_api_sequence_responses = [
        "response Messages object 1",
        "response Messages object 2",
    ]
    mock_anthropic.side_effect = anthropic_api_sequence_responses

    # mock the process_response function
    process_response_sequence_responses = ["response text 1", "response text 2"]
    mock_process_response.side_effect = process_response_sequence_responses

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_chat.keys()

    # call the _query_chat method
    prompt_dict = await anthropic_api._query_chat(prompt_dict_chat, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    assert mock_anthropic.call_count == 2
    assert mock_anthropic.await_count == 2
    mock_anthropic.assert_any_await(
        model=prompt_dict_chat["model_name"],
        messages=[{"role": "user", "content": prompt_dict_chat["prompt"][0]}],
        **prompt_dict_chat["parameters"],
    )
    mock_anthropic.assert_awaited_with(
        model=prompt_dict_chat["model_name"],
        messages=[
            {"role": "user", "content": prompt_dict_chat["prompt"][0]},
            {"role": "assistant", "content": process_response_sequence_responses[0]},
            {"role": "user", "content": prompt_dict_chat["prompt"][1]},
        ],
        **prompt_dict_chat["parameters"],
    )

    assert mock_process_response.call_count == 2
    mock_process_response.assert_any_call(anthropic_api_sequence_responses[0])
    mock_process_response.assert_called_with(anthropic_api_sequence_responses[1])

    # assert that the response value is the return value of the process_response function
    assert prompt_dict["response"] == process_response_sequence_responses

    expected_log_message_1 = (
        f"Response received for model Anthropic ({prompt_dict_chat['model_name']}) "
        "(i=0, id=anthropic_id, message=1/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][0][:50]}...\n"
        f"Response: {process_response_sequence_responses[0][:50]}...\n"
    )
    assert expected_log_message_1 in caplog.text

    expected_log_message_2 = (
        f"Response received for model Anthropic ({prompt_dict_chat['model_name']}) "
        "(i=0, id=anthropic_id, message=2/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][1][:50]}...\n"
        f"Response: {process_response_sequence_responses[1][:50]}...\n"
    )
    assert expected_log_message_2 in caplog.text

    expected_log_message_final = "Chat completed (i=0, id=anthropic_id)"
    assert expected_log_message_final in caplog.text


@pytest.mark.asyncio
@patch("anthropic.resources.AsyncMessages.create", new_callable=CopyingAsyncMock)
async def test_anthropic_query_chat_error_1(
    mock_anthropic, prompt_dict_chat, temporary_data_folders, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "DUMMY")
    anthropic_api = AnthropicAPI(settings=settings, log_file=log_file)

    # mock error response from the API
    mock_anthropic.side_effect = Exception("Test error")

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await anthropic_api._query_chat(prompt_dict_chat, index=0)

    mock_anthropic.assert_called_once()
    mock_anthropic.assert_awaited_once()
    mock_anthropic.assert_any_await(
        model=prompt_dict_chat["model_name"],
        messages=[{"role": "user", "content": prompt_dict_chat["prompt"][0]}],
        **prompt_dict_chat["parameters"],
    )

    expected_log_message = (
        f"Error with model Anthropic ({prompt_dict_chat['model_name']}) "
        "(i=0, id=anthropic_id, message=1/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][0][:50]}...\n"
        "Responses so far: []...\n"
        "Error: Exception - Test error"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch("anthropic.resources.AsyncMessages.create", new_callable=CopyingAsyncMock)
@patch("prompto.apis.anthropic.anthropic.process_response", new_callable=Mock)
async def test_anthropic_query_chat_error_2(
    mock_process_response,
    mock_anthropic,
    prompt_dict_chat,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "DUMMY")
    anthropic_api = AnthropicAPI(settings=settings, log_file=log_file)

    # mock error response from the API from second response
    anthropic_api_sequence_responses = [
        "response Messages object 1",
        Exception("Test error"),
    ]
    mock_anthropic.side_effect = anthropic_api_sequence_responses

    # mock the process_response function
    mock_process_response.return_value = "response text 1"

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await anthropic_api._query_chat(prompt_dict_chat, index=0)

    assert mock_anthropic.call_count == 2
    assert mock_anthropic.await_count == 2
    mock_anthropic.assert_any_await(
        model=prompt_dict_chat["model_name"],
        messages=[{"role": "user", "content": prompt_dict_chat["prompt"][0]}],
        **prompt_dict_chat["parameters"],
    )
    mock_anthropic.assert_awaited_with(
        model=prompt_dict_chat["model_name"],
        messages=[
            {"role": "user", "content": prompt_dict_chat["prompt"][0]},
            {"role": "assistant", "content": mock_process_response.return_value},
            {"role": "user", "content": prompt_dict_chat["prompt"][1]},
        ],
        **prompt_dict_chat["parameters"],
    )

    mock_process_response.assert_called_once_with(anthropic_api_sequence_responses[0])

    expected_log_message_1 = (
        f"Response received for model Anthropic ({prompt_dict_chat['model_name']}) "
        "(i=0, id=anthropic_id, message=1/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][0][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message_1 in caplog.text

    expected_log_message_2 = (
        f"Error with model Anthropic ({prompt_dict_chat['model_name']}) "
        "(i=0, id=anthropic_id, message=2/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][1][:50]}...\n"
        f"Responses so far: {[mock_process_response.return_value]}...\n"
        "Error: Exception - Test error"
    )
    assert expected_log_message_2 in caplog.text
