import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from vertexai.generative_models import GenerativeModel

from prompto.apis.vertexai import VertexAIAPI
from prompto.settings import Settings

from .test_vertexai import (
    DEFAULT_SAFETY_SETTINGS,
    prompt_dict_history,
    prompt_dict_history_no_system,
)

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.ChatSession.send_message_async", new_callable=AsyncMock
)
@patch("prompto.apis.vertexai.vertexai.process_response", new_callable=Mock)
@patch("prompto.apis.vertexai.vertexai.process_safety_attributes", new_callable=Mock)
async def test_vertexai_query_history(
    mock_process_safety_attr,
    mock_process_response,
    mock_vertexai_call,
    prompt_dict_history,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "DUMMY")
    monkeypatch.setenv("VERTEXAI_LOCATION_ID", "europe-west2")
    vertexai_api = VertexAIAPI(settings=settings, log_file=log_file)

    # mock the response from the API
    # NOTE: The actual response from the API is a
    # vertexai.generative_models.types.AsyncGenerateContentResponse object
    # not a string value, but for the purpose of this test, we are using a string value
    # and testing that this is the input to the process_response function
    mock_vertexai_call.return_value = "response Messages object"

    # mock the process_response function
    mock_process_response.return_value = "response text"

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_history.keys()

    # call the _query_history method
    prompt_dict = await vertexai_api._query_history(prompt_dict_history, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "content" is a Content object
    # which is not directly comparable so we convert to dict and compare
    assert "content" in await_kwargs.keys() and await_kwargs["content"].to_dict() == {
        "role": "user",
        "parts": [{"text": prompt_dict_history["prompt"][1]["parts"]}],
    }
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"] == prompt_dict_history["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] == False

    mock_process_response.assert_called_once_with(mock_vertexai_call.return_value)
    mock_process_safety_attr.assert_called_once_with(mock_vertexai_call.return_value)

    # assert that the response value is the return value of the process_response function
    assert prompt_dict["response"] == mock_process_response.return_value

    expected_log_message = (
        f"Response received for model VertexAI ({prompt_dict_history['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_history['prompt'][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.ChatSession.send_message_async", new_callable=AsyncMock
)
async def test_vertexai_query_history_error(
    mock_vertexai_call, prompt_dict_history, temporary_data_folders, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "DUMMY")
    monkeypatch.setenv("VERTEXAI_LOCATION_ID", "europe-west2")
    vertexai_api = VertexAIAPI(settings=settings, log_file=log_file)

    # mock error response from the API
    mock_vertexai_call.side_effect = Exception("Test error")

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await vertexai_api._query_history(prompt_dict_history, index=0)

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "content" is a Content object
    # which is not directly comparable so we convert to dict and compare
    assert "content" in await_kwargs.keys() and await_kwargs["content"].to_dict() == {
        "role": "user",
        "parts": [{"text": prompt_dict_history["prompt"][1]["parts"]}],
    }
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"] == prompt_dict_history["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] == False

    expected_log_message = (
        f"Error with model VertexAI ({prompt_dict_history['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_history['prompt'][:50]}...\n"
        "Error: Exception - Test error"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.ChatSession.send_message_async", new_callable=AsyncMock
)
async def test_vertexai_query_history_index_error(
    mock_vertexai_call, prompt_dict_history, temporary_data_folders, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "DUMMY")
    monkeypatch.setenv("VERTEXAI_LOCATION_ID", "europe-west2")
    vertexai_api = VertexAIAPI(settings=settings, log_file=log_file)

    # mock index error response from the API
    mock_vertexai_call.side_effect = IndexError("Test error")

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_history.keys()

    # call the _query_history method
    prompt_dict = await vertexai_api._query_history(prompt_dict_history, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "content" is a Content object
    # which is not directly comparable so we convert to dict and compare
    assert "content" in await_kwargs.keys() and await_kwargs["content"].to_dict() == {
        "role": "user",
        "parts": [{"text": prompt_dict_history["prompt"][1]["parts"]}],
    }
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"] == prompt_dict_history["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] == False

    expected_log_message = (
        f"Error with model VertexAI ({prompt_dict_history['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_history['prompt'][:50]}...\n"
        "Error: Response is empty and blocked (IndexError - Test error)"
    )
    assert expected_log_message in caplog.text

    # assert that the response value is empty string
    assert prompt_dict["response"] == ""


@pytest.mark.asyncio
@patch("vertexai.generative_models.GenerativeModel.start_chat", new_callable=Mock)
@patch(
    "prompto.apis.vertexai.vertexai.VertexAIAPI._obtain_model_inputs",
    new_callable=AsyncMock,
)
async def test_vertexai_query_history_check_chat_init(
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
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "DUMMY")
    monkeypatch.setenv("VERTEXAI_LOCATION_ID", "europe-west2")
    vertexai_api = VertexAIAPI(settings=settings, log_file=log_file)

    mock_obtain_model_inputs.return_value = (
        prompt_dict_history["prompt"],
        prompt_dict_history["model_name"],
        GenerativeModel(
            model_name=prompt_dict_history["model_name"],
            system_instruction=prompt_dict_history["prompt"][0]["parts"],
        ),
        DEFAULT_SAFETY_SETTINGS,
        prompt_dict_history["parameters"],
        None,
    )

    # error will be raised as we've mocked the start_chat method
    # which leads to an error when the method is called on the mocked object
    with pytest.raises(Exception):
        await vertexai_api._query_history(prompt_dict_history, index=0)

    mock_obtain_model_inputs.assert_called_once_with(
        prompt_dict=prompt_dict_history,
        system_instruction=prompt_dict_history["prompt"][0]["parts"],
    )
    mock_start_chat.assert_called_once_with(history=[])


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.ChatSession.send_message_async", new_callable=AsyncMock
)
@patch("prompto.apis.vertexai.vertexai.process_response", new_callable=Mock)
@patch("prompto.apis.vertexai.vertexai.process_safety_attributes", new_callable=Mock)
async def test_vertexai_query_history_no_system(
    mock_process_safety_attr,
    mock_process_response,
    mock_vertexai_call,
    prompt_dict_history_no_system,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "DUMMY")
    monkeypatch.setenv("VERTEXAI_LOCATION_ID", "europe-west2")
    vertexai_api = VertexAIAPI(settings=settings, log_file=log_file)

    # mock the response from the API
    # NOTE: The actual response from the API is a vertexai.types.message.Message object
    # not a string value, but for the purpose of this test, we are using a string value
    # and testing that this is the input to the process_response function
    mock_vertexai_call.return_value = "response Messages object"

    # mock the process_response function
    mock_process_response.return_value = "response text"

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_history_no_system.keys()

    # call the _query_history method
    prompt_dict = await vertexai_api._query_history(
        prompt_dict_history_no_system, index=0
    )

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "content" is a Content object
    # which is not directly comparable so we convert to dict and compare
    assert "content" in await_kwargs.keys() and await_kwargs["content"].to_dict() == {
        "role": "user",
        "parts": [{"text": prompt_dict_history_no_system["prompt"][2]["parts"]}],
    }
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"]
        == prompt_dict_history_no_system["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] == False

    mock_process_response.assert_called_once_with(mock_vertexai_call.return_value)
    mock_process_safety_attr.assert_called_once_with(mock_vertexai_call.return_value)

    # assert that the response value is the return value of the process_response function
    assert prompt_dict["response"] == mock_process_response.return_value

    expected_log_message = (
        f"Response received for model VertexAI ({prompt_dict_history_no_system['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_history_no_system['prompt'][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.ChatSession.send_message_async", new_callable=AsyncMock
)
async def test_vertexai_query_history_error_no_system(
    mock_vertexai_call,
    prompt_dict_history_no_system,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "DUMMY")
    monkeypatch.setenv("VERTEXAI_LOCATION_ID", "europe-west2")
    vertexai_api = VertexAIAPI(settings=settings, log_file=log_file)

    # mock error response from the API
    mock_vertexai_call.side_effect = Exception("Test error")

    # raise error if the API call fails
    with pytest.raises(Exception, match="Test error"):
        await vertexai_api._query_history(prompt_dict_history_no_system, index=0)

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "content" is a Content object
    # which is not directly comparable so we convert to dict and compare
    assert "content" in await_kwargs.keys() and await_kwargs["content"].to_dict() == {
        "role": "user",
        "parts": [{"text": prompt_dict_history_no_system["prompt"][2]["parts"]}],
    }
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"]
        == prompt_dict_history_no_system["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] == False

    expected_log_message = (
        f"Error with model VertexAI ({prompt_dict_history_no_system['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_history_no_system['prompt'][:50]}...\n"
        "Error: Exception - Test error"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.ChatSession.send_message_async", new_callable=AsyncMock
)
async def test_vertexai_query_history_index_error_no_system(
    mock_vertexai_call,
    prompt_dict_history_no_system,
    temporary_data_folders,
    monkeypatch,
    caplog,
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "DUMMY")
    monkeypatch.setenv("VERTEXAI_LOCATION_ID", "europe-west2")
    vertexai_api = VertexAIAPI(settings=settings, log_file=log_file)

    # mock index error response from the API
    mock_vertexai_call.side_effect = IndexError("Test error")

    # make sure that the input prompt_dict does not have a response key
    assert "response" not in prompt_dict_history_no_system.keys()

    # call the _query_history method
    prompt_dict = await vertexai_api._query_history(
        prompt_dict_history_no_system, index=0
    )

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "content" is a Content object
    # which is not directly comparable so we convert to dict and compare
    assert "content" in await_kwargs.keys() and await_kwargs["content"].to_dict() == {
        "role": "user",
        "parts": [{"text": prompt_dict_history_no_system["prompt"][2]["parts"]}],
    }
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"]
        == prompt_dict_history_no_system["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] == False

    expected_log_message = (
        f"Error with model VertexAI ({prompt_dict_history_no_system['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_history_no_system['prompt'][:50]}...\n"
        "Error: Response is empty and blocked (IndexError - Test error)"
    )
    assert expected_log_message in caplog.text

    # assert that the response value is empty string
    assert prompt_dict["response"] == ""


@pytest.mark.asyncio
@patch("vertexai.generative_models.GenerativeModel.start_chat", new_callable=Mock)
@patch(
    "prompto.apis.vertexai.vertexai.VertexAIAPI._obtain_model_inputs",
    new_callable=AsyncMock,
)
async def test_vertexai_query_history_no_system_check_chat_init(
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
    monkeypatch.setenv("VERTEXAI_PROJECT_ID", "DUMMY")
    monkeypatch.setenv("VERTEXAI_LOCATION_ID", "europe-west2")
    vertexai_api = VertexAIAPI(settings=settings, log_file=log_file)

    mock_obtain_model_inputs.return_value = (
        prompt_dict_history_no_system["prompt"],
        prompt_dict_history_no_system["model_name"],
        GenerativeModel(
            model_name=prompt_dict_history_no_system["model_name"],
            system_instruction=None,
        ),
        DEFAULT_SAFETY_SETTINGS,
        prompt_dict_history_no_system["parameters"],
        None,
    )

    # error will be raised as we've mocked the start_chat method
    # which leads to an error when the method is called on the mocked object
    with pytest.raises(Exception):
        await vertexai_api._query_history(prompt_dict_history_no_system, index=0)

    mock_obtain_model_inputs.assert_called_once_with(
        prompt_dict=prompt_dict_history_no_system, system_instruction=None
    )
    mock_start_chat.assert_called_once()
    await_kwargs = mock_start_chat.call_args.kwargs
    # assert await arguments - slightly more complicated since "history" is a list of Content objects
    # which is not directly comparable so we convert to dict and compare
    assert (
        "history" in await_kwargs.keys()
        and isinstance(await_kwargs["history"], list)
        and len(await_kwargs["history"]) == 2
        and await_kwargs["history"][0].to_dict()
        == {
            "role": "user",
            "parts": [{"text": prompt_dict_history_no_system["prompt"][0]["parts"]}],
        }
        and await_kwargs["history"][1].to_dict()
        == {
            "role": "model",
            "parts": [{"text": prompt_dict_history_no_system["prompt"][1]["parts"]}],
        }
    )
