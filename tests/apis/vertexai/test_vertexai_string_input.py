import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from vertexai.generative_models import Part

from prompto.apis.vertexai import VertexAIAPI
from prompto.settings import Settings

from .test_vertexai import DEFAULT_SAFETY_SETTINGS, prompt_dict_string

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.GenerativeModel.generate_content_async",
    new_callable=AsyncMock,
)
@patch("prompto.apis.vertexai.vertexai.process_response", new_callable=Mock)
@patch("prompto.apis.vertexai.vertexai.process_safety_attributes", new_callable=Mock)
async def test_vertexai_query_string(
    mock_process_safety_attr,
    mock_process_response,
    mock_vertexai_call,
    prompt_dict_string,
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
    assert "response" not in prompt_dict_string.keys()

    # call the _query_string method
    prompt_dict = await vertexai_api._query_string(prompt_dict_string, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "contents" is a
    # list of Part objects which are not directly comparable so we convert to dict and compare
    assert (
        "contents" in await_kwargs.keys()
        and isinstance(await_kwargs["contents"], list)
        and len(await_kwargs["contents"]) == 1
        and await_kwargs["contents"][0].to_dict()
        == {"text": prompt_dict_string["prompt"]}
    )
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"] == prompt_dict_string["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] is False

    mock_process_response.assert_called_once_with(mock_vertexai_call.return_value)
    mock_process_safety_attr.assert_called_once_with(mock_vertexai_call.return_value)

    # assert that the response value is the return value of the process_response function
    assert prompt_dict["response"] == mock_process_response.return_value

    expected_log_message = (
        f"Response received for model VertexAI ({prompt_dict_string['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_string['prompt'][:50]}...\n"
        f"Response: {mock_process_response.return_value[:50]}...\n"
    )
    assert expected_log_message in caplog.text


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.GenerativeModel.generate_content_async",
    new_callable=AsyncMock,
)
async def test_vertexai_query_string__index_error(
    mock_vertexai_call, prompt_dict_string, temporary_data_folders, monkeypatch, caplog
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
    assert "response" not in prompt_dict_string.keys()

    # call the _query_string method
    prompt_dict = await vertexai_api._query_string(prompt_dict_string, index=0)

    # assert that the response key is added to the prompt_dict
    assert "response" in prompt_dict.keys()

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "contents" is a
    # list of Part objects which are not directly comparable so we convert to dict and compare
    assert (
        "contents" in await_kwargs.keys()
        and isinstance(await_kwargs["contents"], list)
        and len(await_kwargs["contents"]) == 1
        and await_kwargs["contents"][0].to_dict()
        == {"text": prompt_dict_string["prompt"]}
    )
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"] == prompt_dict_string["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] is False

    expected_log_message = (
        f"Error with model VertexAI ({prompt_dict_string['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_string['prompt'][:50]}...\n"
        "Error: Response is empty and blocked (IndexError - Test error)\n"
    )
    assert expected_log_message in caplog.text

    # assert that the response value is empty string
    assert prompt_dict["response"] == ""


@pytest.mark.asyncio
@patch(
    "vertexai.generative_models.GenerativeModel.generate_content_async",
    new_callable=AsyncMock,
)
async def test_vertexai_query_string_error(
    mock_vertexai_call, prompt_dict_string, temporary_data_folders, monkeypatch, caplog
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
        await vertexai_api._query_string(prompt_dict_string, index=0)

    mock_vertexai_call.assert_called_once()
    mock_vertexai_call.assert_awaited_once()
    await_kwargs = mock_vertexai_call.await_args.kwargs
    # assert await arguments - slightly more complicated since "contents" is a
    # list of Part objects which are not directly comparable so we convert to dict and compare
    assert (
        "contents" in await_kwargs.keys()
        and isinstance(await_kwargs["contents"], list)
        and len(await_kwargs["contents"]) == 1
        and await_kwargs["contents"][0].to_dict()
        == {"text": prompt_dict_string["prompt"]}
    )
    assert (
        "generation_config" in await_kwargs.keys()
        and await_kwargs["generation_config"] == prompt_dict_string["parameters"]
    )
    assert (
        "safety_settings" in await_kwargs.keys()
        and await_kwargs["safety_settings"] == DEFAULT_SAFETY_SETTINGS
    )
    assert "stream" in await_kwargs.keys() and await_kwargs["stream"] is False

    expected_log_message = (
        f"Error with model VertexAI ({prompt_dict_string['model_name']}) "
        "(i=0, id=vertexai_id)\n"
        f"Prompt: {prompt_dict_string['prompt'][:50]}...\n"
        "Error: Exception - Test error\n"
    )
    assert expected_log_message in caplog.text
