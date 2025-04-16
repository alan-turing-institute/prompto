import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from prompto.apis.anthropic import AnthropicAPI
from prompto.settings import Settings

from .test_anthropic import prompt_dict_string

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_anthropic_query_string_no_env_var(
    prompt_dict_string, temporary_data_folders, caplog
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
        await anthropic_api._query_string(prompt_dict_string, index=0)


@pytest.mark.asyncio
@patch("anthropic.resources.AsyncMessages.create", new_callable=AsyncMock)
@patch("prompto.apis.anthropic.anthropic.process_response", new_callable=Mock)
async def test_anthropic_query_string(
    mock_process_response,
    mock_anthropic,
    prompt_dict_string,
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
    # NOTE: The actual response from the API is a anthropic.types.message.Message object
    # not a string value, but for the purpose of this test, we are using a string value
    # and testing that this is the input to the process_response function
    mock_anthropic.return_value = "response Messages object"

    # mock the process_response function
    mock_process_response.return_value = "response text"

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_string.keys()

    # call the _query_string method
    prompt_dict = await anthropic_api._query_string(prompt_dict_string, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_anthropic.assert_called_once()
    mock_anthropic.assert_awaited_once()
    mock_anthropic.assert_awaited_once_with(
        model=prompt_dict_string["model_name"],
        messages=[{"role": "user", "content": prompt_dict_string["prompt"]}],
        **prompt_dict_string["parameters"],
    )

    mock_process_response.assert_called_once_with(mock_anthropic.return_value)

    # assert that the response value is the return value of the process_response function
    assert prompt_dict["response"] == mock_process_response.return_value

    expected_log_message = (
        f"Response received for model Anthropic ({prompt_dict_string['model_name']}) "
        "(i=0, id=anthropic_id)\n"
        f"Prompt: {prompt_dict_string['prompt'][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch("anthropic.resources.AsyncMessages.create", new_callable=AsyncMock)
async def test_anthropic_query_string_error(
    mock_anthropic, prompt_dict_string, temporary_data_folders, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("ANTHROPIC_API_KEY_anthropic_model_name", "DUMMY")
    anthropic_api = AnthropicAPI(settings=settings, log_file=log_file)

    # mock error response from the API
    mock_anthropic.side_effect = Exception("Test error")

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await anthropic_api._query_string(prompt_dict_string, index=0)

    mock_anthropic.assert_called_once()
    mock_anthropic.assert_awaited_once()
    mock_anthropic.assert_awaited_once_with(
        model=prompt_dict_string["model_name"],
        messages=[{"role": "user", "content": prompt_dict_string["prompt"]}],
        **prompt_dict_string["parameters"],
    )

    expected_log_message = (
        f"Error with model Anthropic ({prompt_dict_string['model_name']}) "
        "(i=0, id=anthropic_id)\n"
        f"Prompt: {prompt_dict_string['prompt'][:50]}...\n"
        "Error: Exception - Test error\n"
    )
    assert expected_log_message in caplog.text
