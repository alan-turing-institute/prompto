import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

# from google.generativeai import GenerativeModel
from google.genai.chats import AsyncChats, Chat

from prompto.apis.gemini import GeminiAPI
from prompto.settings import Settings

from ...conftest import CopyingAsyncMock
from .test_gemini import DEFAULT_SAFETY_SETTINGS, prompt_dict_chat

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_gemini_query_chat_no_env_var(
    prompt_dict_chat, temporary_data_folders, caplog
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
        await gemini_api._query_chat(prompt_dict_chat, index=0)


@pytest.mark.asyncio
# @patch(
#     "google.generativeai.ChatSession.send_message_async", new_callable=CopyingAsyncMock
# )
@patch.object(
    Chat,
    "send_message",
    new_callable=CopyingAsyncMock,
)
@patch("prompto.apis.gemini.gemini.process_response", new_callable=Mock)
@patch("prompto.apis.gemini.gemini.process_safety_attributes", new_callable=Mock)
async def test_gemini_query_chat(
    mock_process_safety_attr,
    mock_process_response,
    mock_gemini_call,
    prompt_dict_chat,
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
    gemini_api_sequence_responses = [
        "response Messages object 1",
        "response Messages object 2",
    ]
    mock_gemini_call.side_effect = gemini_api_sequence_responses

    # mock the process_response function
    process_response_sequence_responses = ["response text 1", "response text 2"]
    mock_process_response.side_effect = process_response_sequence_responses

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_chat.keys()

    # call the _query_chat method
    prompt_dict = await gemini_api._query_chat(prompt_dict_chat, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    assert mock_gemini_call.call_count == 2
    assert mock_gemini_call.await_count == 2
    mock_gemini_call.assert_any_await(
        content=prompt_dict_chat["prompt"][0],
        generation_config=prompt_dict_chat["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )
    mock_gemini_call.assert_awaited_with(
        content=prompt_dict_chat["prompt"][1],
        generation_config=prompt_dict_chat["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    assert mock_process_response.call_count == 2
    mock_process_response.assert_any_call(gemini_api_sequence_responses[0])
    mock_process_response.assert_called_with(gemini_api_sequence_responses[1])

    assert mock_process_safety_attr.call_count == 2
    mock_process_safety_attr.assert_any_call(gemini_api_sequence_responses[0])
    mock_process_safety_attr.assert_called_with(gemini_api_sequence_responses[1])

    # assert that the response value is the return value of the process_response function
    assert prompt_dict["response"] == process_response_sequence_responses

    expected_log_message_1 = (
        f"Response received for model Gemini ({prompt_dict_chat['model_name']}) "
        "(i=0, id=gemini_id, message=1/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][0][:50]}...\n"
        f"Response: {process_response_sequence_responses[0][:50]}...\n"
    )
    assert expected_log_message_1 in caplog.text

    expected_log_message_2 = (
        f"Response received for model Gemini ({prompt_dict_chat['model_name']}) "
        "(i=0, id=gemini_id, message=2/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][1][:50]}...\n"
        f"Response: {process_response_sequence_responses[1][:50]}...\n"
    )
    assert expected_log_message_2 in caplog.text

    expected_log_message_final = "Chat completed (i=0, id=gemini_id)"
    assert expected_log_message_final in caplog.text


@pytest.mark.asyncio
# @patch("google.generativeai.GenerativeModel.start_chat", new_callable=Mock)
@patch.object(
    AsyncChats,
    "create",
    new_callable=AsyncMock,
)
@patch(
    "prompto.apis.gemini.gemini.GeminiAPI._obtain_model_inputs", new_callable=AsyncMock
)
async def test_gemini_query_history_check_chat_init(
    mock_obtain_model_inputs,
    mock_start_chat,
    prompt_dict_chat,
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
        prompt_dict_chat["prompt"],
        prompt_dict_chat["model_name"],
        GenerativeModel(
            model_name=prompt_dict_chat["model_name"], system_instruction=None
        ),
        DEFAULT_SAFETY_SETTINGS,
        prompt_dict_chat["parameters"],
    )

    # error will be raised as we've mocked the start_chat method
    # which leads to an error when the method is called on the mocked object
    with pytest.raises(Exception):
        await gemini_api._query_chat(prompt_dict_chat, index=0)

    mock_obtain_model_inputs.assert_called_once_with(
        prompt_dict=prompt_dict_chat, system_instruction=None
    )
    mock_start_chat.assert_called_once_with(history=[])


@pytest.mark.asyncio
# @patch(
#     "google.generativeai.ChatSession.send_message_async", new_callable=CopyingAsyncMock
# )
@patch.object(
    Chat,
    "send_message",
    new_callable=CopyingAsyncMock,
)
async def test_gemini_query_chat_index_error_1(
    mock_gemini_call, prompt_dict_chat, temporary_data_folders, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock index error response from the API
    mock_gemini_call.side_effect = IndexError("Test error")

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_chat.keys()

    # call the _query_chat method
    prompt_dict = await gemini_api._query_chat(prompt_dict_chat, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_gemini_call.assert_called_once()
    mock_gemini_call.assert_awaited_once()
    mock_gemini_call.assert_any_await(
        content=prompt_dict_chat["prompt"][0],
        generation_config=prompt_dict_chat["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    expected_log_message = (
        f"Error with model Gemini ({prompt_dict_chat['model_name']}) "
        "(i=0, id=gemini_id, message=1/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][0][:50]}...\n"
        "Responses so far: []...\n"
        "Error: Response is empty and blocked (IndexError - Test error)"
    )
    assert expected_log_message in caplog.text

    # assert that the response value is empty string
    assert prompt_dict["response"] == [""]


@pytest.mark.asyncio
# @patch(
#     "google.generativeai.ChatSession.send_message_async", new_callable=CopyingAsyncMock
# )
@patch.object(
    Chat,
    "send_message",
    new_callable=CopyingAsyncMock,
)
async def test_gemini_query_chat_error_1(
    mock_gemini_call, prompt_dict_chat, temporary_data_folders, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock error response from the API
    mock_gemini_call.side_effect = Exception("Test error")

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await gemini_api._query_chat(prompt_dict_chat, index=0)

    mock_gemini_call.assert_called_once()
    mock_gemini_call.assert_awaited_once()
    mock_gemini_call.assert_any_await(
        content=prompt_dict_chat["prompt"][0],
        generation_config=prompt_dict_chat["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    expected_log_message = (
        f"Error with model Gemini ({prompt_dict_chat['model_name']}) "
        "(i=0, id=gemini_id, message=1/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][0][:50]}...\n"
        "Responses so far: []...\n"
        "Error: Exception - Test error"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
# @patch(
#     "google.generativeai.ChatSession.send_message_async", new_callable=CopyingAsyncMock
# )
@patch.object(
    Chat,
    "send_message",
    new_callable=CopyingAsyncMock,
)
@patch("prompto.apis.gemini.gemini.process_response", new_callable=Mock)
@patch("prompto.apis.gemini.gemini.process_safety_attributes", new_callable=Mock)
async def test_gemini_query_chat_index_error_2(
    mock_process_safety_attr,
    mock_process_response,
    mock_gemini_call,
    prompt_dict_chat,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock error response from the API from second response
    gemini_api_sequence_responses = [
        "response Messages object 1",
        IndexError("Test error"),
    ]

    mock_gemini_call.side_effect = gemini_api_sequence_responses

    # mock the process_response function
    mock_process_response.return_value = "response text 1"

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_chat.keys()

    # call the _query_chat method
    prompt_dict = await gemini_api._query_chat(prompt_dict_chat, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    assert mock_gemini_call.call_count == 2
    assert mock_gemini_call.await_count == 2
    mock_gemini_call.assert_any_await(
        content=prompt_dict_chat["prompt"][0],
        generation_config=prompt_dict_chat["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )
    mock_gemini_call.assert_awaited_with(
        content=prompt_dict_chat["prompt"][1],
        generation_config=prompt_dict_chat["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    mock_process_response.assert_called_once_with(gemini_api_sequence_responses[0])
    mock_process_safety_attr.assert_called_once_with(gemini_api_sequence_responses[0])

    expected_log_message_1 = (
        f"Response received for model Gemini ({prompt_dict_chat['model_name']}) "
        "(i=0, id=gemini_id, message=1/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][0][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message_1 in caplog.text

    expected_log_message_2 = (
        f"Error with model Gemini ({prompt_dict_chat['model_name']}) "
        "(i=0, id=gemini_id, message=2/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][1][:50]}...\n"
        f"Responses so far: {[mock_process_response.return_value]}...\n"
        "Error: Response is empty and blocked (IndexError - Test error)"
    )
    assert expected_log_message_2 in caplog.text

    # assert that the response value is the first response value and an empty string
    assert prompt_dict["response"] == [mock_process_response.return_value, ""]


@pytest.mark.asyncio
# @patch(
#     "google.generativeai.ChatSession.send_message_async", new_callable=CopyingAsyncMock
# )
@patch.object(
    Chat,
    "send_message",
    new_callable=CopyingAsyncMock,
)
@patch("prompto.apis.gemini.gemini.process_response", new_callable=Mock)
@patch("prompto.apis.gemini.gemini.process_safety_attributes", new_callable=Mock)
async def test_gemini_query_chat_error_2(
    mock_process_safety_attr,
    mock_process_response,
    mock_gemini_call,
    prompt_dict_chat,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock error response from the API from second response
    gemini_api_sequence_responses = [
        "response Messages object 1",
        Exception("Test error"),
    ]
    mock_gemini_call.side_effect = gemini_api_sequence_responses

    # mock the process_response function
    mock_process_response.return_value = "response text 1"

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await gemini_api._query_chat(prompt_dict_chat, index=0)

    assert mock_gemini_call.call_count == 2
    assert mock_gemini_call.await_count == 2
    mock_gemini_call.assert_any_await(
        content=prompt_dict_chat["prompt"][0],
        generation_config=prompt_dict_chat["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )
    mock_gemini_call.assert_awaited_with(
        content=prompt_dict_chat["prompt"][1],
        generation_config=prompt_dict_chat["parameters"],
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        stream=False,
    )

    mock_process_response.assert_called_once_with(gemini_api_sequence_responses[0])
    mock_process_safety_attr.assert_called_once_with(gemini_api_sequence_responses[0])

    expected_log_message_1 = (
        f"Response received for model Gemini ({prompt_dict_chat['model_name']}) "
        "(i=0, id=gemini_id, message=1/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][0][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message_1 in caplog.text

    expected_log_message_2 = (
        f"Error with model Gemini ({prompt_dict_chat['model_name']}) "
        "(i=0, id=gemini_id, message=2/2)\n"
        f"Prompt: {prompt_dict_chat['prompt'][1][:50]}...\n"
        f"Responses so far: {[mock_process_response.return_value]}...\n"
        "Error: Exception - Test error"
    )
    assert expected_log_message_2 in caplog.text
