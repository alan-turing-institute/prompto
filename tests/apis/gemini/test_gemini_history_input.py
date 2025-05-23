import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.generativeai import GenerativeModel

from prompto.apis.gemini import GeminiAPI
from prompto.settings import Settings

from .test_gemini import (
    DEFAULT_SAFETY_SETTINGS,
    prompt_dict_history,
    prompt_dict_history_no_system,
)

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_gemini_query_history_no_env_var(
    prompt_dict_history, temporary_data_folders, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # raise error if no environment variable is set
    with pytest.raises(
        KeyError,
        match=(
            "Neither 'GEMINI_API_KEY' nor 'GEMINI_API_KEY_gemini_model_name' "
            "environment variable is set."
        ),
    ):
        await gemini_api._query_history(prompt_dict_history, index=0)


@pytest.mark.asyncio
@patch("google.generativeai.ChatSession.send_message_async", new_callable=AsyncMock)
@patch("prompto.apis.gemini.gemini.process_response", new_callable=Mock)
@patch("prompto.apis.gemini.gemini.process_safety_attributes", new_callable=Mock)
async def test_gemini_query_history(
    mock_process_safety_attr,
    mock_process_response,
    mock_gemini_call,
    prompt_dict_history,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock the response from the API
    # NOTE: The actual response from the API is a
    # google.generativeai.types.AsyncGenerateContentResponse object
    # not a string value, but for the purpose of this test, we are using a string value
    # and testing that this is the input to the process_response function
    mock_gemini_call.return_value = "response Messages object"

    # mock the process_response function
    mock_process_response.return_value = "response text"

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_history.keys()

    # call the _query_history method
    prompt_dict = await gemini_api._query_history(prompt_dict_history, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_gemini_call.assert_called_once()
    mock_gemini_call.assert_awaited_once()
    mock_gemini_call.assert_awaited_once_with(
        content={"role": "user", "parts": [prompt_dict_history["prompt"][1]["parts"]]},
        generation_config=prompt_dict_history["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    mock_process_response.assert_called_once_with(mock_gemini_call.return_value)
    mock_process_safety_attr.assert_called_once_with(mock_gemini_call.return_value)

    # assert that the response value is the return value of the process_response function
    assert prompt_dict["response"] == mock_process_response.return_value

    expected_log_message = (
        f"Response received for model Gemini ({prompt_dict_history['model_name']}) "
        "(i=0, id=gemini_id)\n"
        f"Prompt: {prompt_dict_history['prompt'][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch("google.generativeai.ChatSession.send_message_async", new_callable=AsyncMock)
async def test_gemini_query_history_error(
    mock_gemini_call, prompt_dict_history, temporary_data_folders, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY_gemini_model_name", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock error response from the API
    mock_gemini_call.side_effect = Exception("Test error")

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await gemini_api._query_history(prompt_dict_history, index=0)

    mock_gemini_call.assert_called_once()
    mock_gemini_call.assert_awaited_once()
    mock_gemini_call.assert_awaited_once_with(
        content={"role": "user", "parts": [prompt_dict_history["prompt"][1]["parts"]]},
        generation_config=prompt_dict_history["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    expected_log_message = (
        f"Error with model Gemini ({prompt_dict_history['model_name']}) "
        "(i=0, id=gemini_id)\n"
        f"Prompt: {prompt_dict_history['prompt'][:50]}...\n"
        "Error: Exception - Test error"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch("google.generativeai.ChatSession.send_message_async", new_callable=AsyncMock)
async def test_gemini_query_history_index_error(
    mock_gemini_call, prompt_dict_history, temporary_data_folders, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY_gemini_model_name", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock index error response from the API
    mock_gemini_call.side_effect = IndexError("Test error")

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_history.keys()

    # call the _query_history method
    prompt_dict = await gemini_api._query_history(prompt_dict_history, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_gemini_call.assert_called_once()
    mock_gemini_call.assert_awaited_once()
    mock_gemini_call.assert_awaited_once_with(
        content={"role": "user", "parts": [prompt_dict_history["prompt"][1]["parts"]]},
        generation_config=prompt_dict_history["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    expected_log_message = (
        f"Error with model Gemini ({prompt_dict_history['model_name']}) "
        "(i=0, id=gemini_id)\n"
        f"Prompt: {prompt_dict_history['prompt'][:50]}...\n"
        "Error: Response is empty and blocked (IndexError - Test error)"
    )
    assert expected_log_message in caplog.text

    # assert that the response value is empty string
    assert prompt_dict["response"] == ""


@pytest.mark.asyncio
@patch("google.generativeai.GenerativeModel.start_chat", new_callable=Mock)
@patch(
    "prompto.apis.gemini.gemini.GeminiAPI._obtain_model_inputs", new_callable=AsyncMock
)
async def test_gemini_query_history_check_chat_init(
    mock_obtain_model_inputs,
    mock_start_chat,
    prompt_dict_history,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY_gemini_model_name", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    mock_obtain_model_inputs.return_value = (
        prompt_dict_history["prompt"],
        prompt_dict_history["model_name"],
        GenerativeModel(
            model_name=prompt_dict_history["model_name"],
            system_instruction=prompt_dict_history["prompt"][0]["parts"],
        ),
        DEFAULT_SAFETY_SETTINGS,
        prompt_dict_history["parameters"],
    )

    # error will be raised as we've mocked the start_chat method
    # which leads to an error when the method is called on the mocked object
    with pytest.raises(Exception):
        await gemini_api._query_history(prompt_dict_history, index=0)

    mock_obtain_model_inputs.assert_called_once_with(
        prompt_dict=prompt_dict_history,
        system_instruction=prompt_dict_history["prompt"][0]["parts"],
    )
    mock_start_chat.assert_called_once_with(history=[])


@pytest.mark.asyncio
@patch("google.generativeai.ChatSession.send_message_async", new_callable=AsyncMock)
@patch("prompto.apis.gemini.gemini.process_response", new_callable=Mock)
@patch("prompto.apis.gemini.gemini.process_safety_attributes", new_callable=Mock)
async def test_gemini_query_history_no_system(
    mock_process_safety_attr,
    mock_process_response,
    mock_gemini_call,
    prompt_dict_history_no_system,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock the response from the API
    # NOTE: The actual response from the API is a gemini.types.message.Message object
    # not a string value, but for the purpose of this test, we are using a string value
    # and testing that this is the input to the process_response function
    mock_gemini_call.return_value = "response Messages object"

    # mock the process_response function
    mock_process_response.return_value = "response text"

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_history_no_system.keys()

    # call the _query_history method
    prompt_dict = await gemini_api._query_history(
        prompt_dict_history_no_system, index=0
    )

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_gemini_call.assert_called_once()
    mock_gemini_call.assert_awaited_once()
    mock_gemini_call.assert_awaited_once_with(
        content={
            "role": "user",
            "parts": [prompt_dict_history_no_system["prompt"][2]["parts"]],
        },
        generation_config=prompt_dict_history_no_system["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    mock_process_response.assert_called_once_with(mock_gemini_call.return_value)
    mock_process_safety_attr.assert_called_once_with(mock_gemini_call.return_value)

    # assert that the response value is the return value of the process_response function
    assert prompt_dict["response"] == mock_process_response.return_value

    expected_log_message = (
        f"Response received for model Gemini ({prompt_dict_history_no_system['model_name']}) "
        "(i=0, id=gemini_id)\n"
        f"Prompt: {prompt_dict_history_no_system['prompt'][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch("google.generativeai.ChatSession.send_message_async", new_callable=AsyncMock)
async def test_gemini_query_history_error_no_system(
    mock_gemini_call,
    prompt_dict_history_no_system,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY_gemini_model_name", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock error response from the API
    mock_gemini_call.side_effect = Exception("Test error")

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await gemini_api._query_history(prompt_dict_history_no_system, index=0)

    mock_gemini_call.assert_called_once()
    mock_gemini_call.assert_awaited_once()
    mock_gemini_call.assert_awaited_once_with(
        content={
            "role": "user",
            "parts": [prompt_dict_history_no_system["prompt"][2]["parts"]],
        },
        generation_config=prompt_dict_history_no_system["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    expected_log_message = (
        f"Error with model Gemini ({prompt_dict_history_no_system['model_name']}) "
        "(i=0, id=gemini_id)\n"
        f"Prompt: {prompt_dict_history_no_system['prompt'][:50]}...\n"
        "Error: Exception - Test error"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch("google.generativeai.ChatSession.send_message_async", new_callable=AsyncMock)
async def test_gemini_query_history_index_error_no_system(
    mock_gemini_call,
    prompt_dict_history_no_system,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY_gemini_model_name", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock index error response from the API
    mock_gemini_call.side_effect = IndexError("Test error")

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_history_no_system.keys()

    # call the _query_history method
    prompt_dict = await gemini_api._query_history(
        prompt_dict_history_no_system, index=0
    )

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_gemini_call.assert_called_once()
    mock_gemini_call.assert_awaited_once()
    mock_gemini_call.assert_awaited_once_with(
        content={
            "role": "user",
            "parts": [prompt_dict_history_no_system["prompt"][2]["parts"]],
        },
        generation_config=prompt_dict_history_no_system["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    expected_log_message = (
        f"Error with model Gemini ({prompt_dict_history_no_system['model_name']}) "
        "(i=0, id=gemini_id)\n"
        f"Prompt: {prompt_dict_history_no_system['prompt'][:50]}...\n"
        "Error: Response is empty and blocked (IndexError - Test error)"
    )
    assert expected_log_message in caplog.text

    # assert that the response value is empty string
    assert prompt_dict["response"] == ""


@pytest.mark.asyncio
@patch("google.generativeai.GenerativeModel.start_chat", new_callable=Mock)
@patch(
    "prompto.apis.gemini.gemini.GeminiAPI._obtain_model_inputs", new_callable=AsyncMock
)
async def test_gemini_query_history_no_system_check_chat_init(
    mock_obtain_model_inputs,
    mock_start_chat,
    prompt_dict_history_no_system,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY_gemini_model_name", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    mock_obtain_model_inputs.return_value = (
        prompt_dict_history_no_system["prompt"],
        prompt_dict_history_no_system["model_name"],
        GenerativeModel(
            model_name=prompt_dict_history_no_system["model_name"],
            system_instruction=None,
        ),
        DEFAULT_SAFETY_SETTINGS,
        prompt_dict_history_no_system["parameters"],
    )

    # error will be raised as we've mocked the start_chat method
    # which leads to an error when the method is called on the mocked object
    with pytest.raises(Exception):
        await gemini_api._query_history(prompt_dict_history_no_system, index=0)

    mock_obtain_model_inputs.assert_called_once_with(
        prompt_dict=prompt_dict_history_no_system, system_instruction=None
    )
    mock_start_chat.assert_called_once_with(
        history=[
            {
                "role": "user",
                "parts": [prompt_dict_history_no_system["prompt"][0]["parts"]],
            },
            {
                "role": "model",
                "parts": [prompt_dict_history_no_system["prompt"][1]["parts"]],
            },
        ]
    )
