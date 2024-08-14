from unittest.mock import AsyncMock, patch

import pytest
import regex as re
from google.generativeai import GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory, content_types

from prompto.apis.gemini import GeminiAPI
from prompto.settings import Settings

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def prompt_dict_string():
    return {
        "id": "gemini_id",
        "api": "gemini",
        "model_name": "gemini_model_name",
        "prompt": "test prompt",
        "parameters": {"temperature": 1, "max_output_tokens": 100},
    }


@pytest.fixture
def prompt_dict_chat():
    return {
        "id": "gemini_id",
        "api": "gemini",
        "model_name": "gemini_model_name",
        "prompt": ["test chat 1", "test chat 2"],
        "parameters": {"temperature": 1, "max_output_tokens": 100},
    }


@pytest.fixture
def prompt_dict_history():
    return {
        "id": "gemini_id",
        "api": "gemini",
        "model_name": "gemini_model_name",
        "prompt": [
            {"role": "system", "parts": "test system prompt"},
            {"role": "user", "parts": "user message"},
        ],
        "parameters": {"temperature": 1, "max_output_tokens": 100},
    }


@pytest.fixture
def prompt_dict_history_no_system():
    return {
        "id": "gemini_id",
        "api": "gemini",
        "model_name": "gemini_model_name",
        "prompt": [
            {"role": "user", "parts": "user message 1"},
            {"role": "model", "parts": "model message"},
            {"role": "user", "parts": "user message 2"},
        ],
        "parameters": {"temperature": 1, "max_output_tokens": 100},
    }


DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

TYPE_ERROR_MSG = (
    "if api == 'gemini', then the prompt must be a str, list[str], or "
    "list[dict[str,str]] where the dictionary contains the keys 'role' and "
    "'parts' only, and the values for 'role' must be one of 'user' or 'model', "
    "except for the first message in the list of dictionaries can be a "
    "system message with the key 'role' set to 'system'."
)


def test_gemini_api_init(temporary_data_folders):
    # raise error if no arguments are provided
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        GeminiAPI()

    # raise error if no settings object is provided
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'settings'"
    ):
        GeminiAPI(log_file="log.txt")

    # raise error if no log file name is provided
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'log_file'"
    ):
        GeminiAPI(settings=Settings(data_folder="data"))

    settings = Settings(data_folder="data")
    log_file = "log.txt"
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    assert gemini_api.settings == settings
    assert gemini_api.log_file == log_file


def test_gemini_check_environment_variables(temporary_data_folders, monkeypatch):
    # only warn on the GEMINI_API_KEY environment variable
    test_case = GeminiAPI.check_environment_variables()
    assert len(test_case) == 1
    with pytest.raises(
        Warning, match="Environment variable 'GEMINI_API_KEY' is not set"
    ):
        raise test_case[0]

    # no warnings or errors
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    assert GeminiAPI.check_environment_variables() == []


def test_gemini_check_prompt_dict(temporary_data_folders, monkeypatch):
    env_variable_error_msg = (
        "At least one of the environment variables '['GEMINI_API_KEY_gemini_model_name', "
        "'GEMINI_API_KEY']' must be set"
    )

    # error if prompt_dict["prompt"] is not of the correct type
    # also error for no environment variables set
    test_case = GeminiAPI.check_prompt_dict(
        {"api": "gemini", "model_name": "gemini_model_name", "prompt": 1}
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        raise test_case[0]
    with pytest.raises(KeyError, match=re.escape(env_variable_error_msg)):
        raise test_case[1]

    # error if prompt_dict["prompt"] is a list of strings but
    # also contains an incorrect type
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": ["prompt 1", "prompt 2", 1],
        }
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        raise test_case[0]

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # also contains an incorrect type
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": [
                {"role": "system", "parts": "system prompt"},
                {"role": "user", "parts": "user message"},
                1,
            ],
        }
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        raise test_case[0]

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # one of the roles is incorrect
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": [
                {"role": "system", "parts": "system prompt"},
                {"role": "user", "parts": "user message"},
                {"role": "incorrect", "parts": "some message"},
            ],
        }
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        raise test_case[0]

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # the system message is not the first message in the list
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": [
                {"role": "user", "parts": "user message"},
                {"role": "system", "parts": "system prompt"},
            ],
        }
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        raise test_case[0]

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # there are multiple system messages in the list
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": [
                {"role": "system", "parts": "system prompt 1"},
                {"role": "system", "parts": "system prompt 2"},
                {"role": "user", "parts": "user message"},
            ],
        }
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        raise test_case[0]

    # error if safety_filter is not recognised
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": "test prompt",
            "safety_filter": "unknown-safety-filter",
        }
    )
    assert len(test_case) == 2
    with pytest.raises(
        ValueError,
        match="Invalid safety_filter value",
    ):
        raise test_case[1]

    # error if there is a parameter that is not recognised by Gemini
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": "test prompt",
            "parameters": {"temperature": 1, "max_output_tokens": 100, "unknown": 1},
        }
    )
    assert len(test_case) == 2
    with pytest.raises(Exception, match="Invalid generation_config parameter: "):
        raise test_case[1]

    # error if neither environment variable is set
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": "test prompt",
        }
    )
    assert len(test_case) == 1
    with pytest.raises(KeyError, match=re.escape(env_variable_error_msg)):
        raise test_case[0]

    # set the GEMINI_API_KEY environment variable
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    # error if the model-specific environment variable is not set
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": "test prompt",
        }
    )
    assert len(test_case) == 1
    with pytest.raises(
        Warning,
        match=re.escape(
            "Environment variable 'GEMINI_API_KEY_gemini_model_name' is not set"
        ),
    ):
        raise test_case[0]

    # unset the GEMINI_API_KEY environment variable and
    # set the model-specific environment variable
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY_gemini_model_name", "DUMMY")
    test_case = GeminiAPI.check_prompt_dict(
        {
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": "test prompt",
        }
    )
    assert len(test_case) == 1
    with pytest.raises(
        Warning, match=re.escape("Environment variable 'GEMINI_API_KEY' is not set")
    ):
        raise test_case[0]

    # full passes
    # set both environment variables
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    assert (
        GeminiAPI.check_prompt_dict(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": "test prompt",
            }
        )
        == []
    )
    assert (
        GeminiAPI.check_prompt_dict(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": ["prompt 1", "prompt 2"],
            }
        )
        == []
    )
    assert (
        GeminiAPI.check_prompt_dict(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": [
                    {"role": "system", "parts": "system prompt"},
                    {"role": "user", "parts": "user message 1"},
                    {"role": "model", "parts": "model message"},
                    {"role": "user", "parts": "user message 2"},
                ],
            }
        )
        == []
    )
    assert (
        GeminiAPI.check_prompt_dict(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": [
                    {"role": "user", "parts": "user message 1"},
                    {"role": "model", "parts": "model message"},
                    {"role": "user", "parts": "user message 2"},
                ],
            }
        )
        == []
    )


@pytest.mark.asyncio
async def test_gemini_obtain_model_inputs(temporary_data_folders, monkeypatch):
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # test for string prompt
    test_case = await gemini_api._obtain_model_inputs(
        {
            "id": "gemini_id",
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": "test prompt",
            "parameters": {"temperature": 1, "max_output_tokens": 100},
        }
    )
    assert isinstance(test_case, tuple)
    assert len(test_case) == 6
    assert test_case[0] == "test prompt"
    assert test_case[1] == "gemini_model_name"
    assert isinstance(test_case[2], GenerativeModel)
    assert test_case[2]._model_name == "models/gemini_model_name"
    assert test_case[2]._system_instruction is None
    assert isinstance(test_case[3], dict)
    assert test_case[4] == {"temperature": 1, "max_output_tokens": 100}
    assert test_case[5] is None

    # test for case where no parameters in prompt_dict
    test_case = await gemini_api._obtain_model_inputs(
        {
            "id": "gemini_id",
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": "test prompt",
        }
    )
    assert isinstance(test_case, tuple)
    assert len(test_case) == 6
    assert test_case[0] == "test prompt"
    assert test_case[1] == "gemini_model_name"
    assert isinstance(test_case[2], GenerativeModel)
    assert test_case[2]._model_name == "models/gemini_model_name"
    assert test_case[2]._system_instruction is None
    assert isinstance(test_case[3], dict)
    assert test_case[4] == {}
    assert test_case[5] is None

    # test for case where system_instruction is provided
    test_case = await gemini_api._obtain_model_inputs(
        {
            "id": "gemini_id",
            "api": "gemini",
            "model_name": "gemini_model_name",
            "prompt": "test prompt",
        },
        system_instruction="hello",
    )
    assert isinstance(test_case, tuple)
    assert len(test_case) == 6
    assert test_case[0] == "test prompt"
    assert test_case[1] == "gemini_model_name"
    assert isinstance(test_case[2], GenerativeModel)
    assert test_case[2]._model_name == "models/gemini_model_name"
    assert test_case[2]._system_instruction is not None
    assert isinstance(test_case[3], dict)
    assert test_case[4] == {}
    assert test_case[5] is None

    # test error catching when parameters are not a dictionary
    with pytest.raises(
        TypeError, match="parameters must be a dictionary, not <class 'int'>"
    ):
        await gemini_api._obtain_model_inputs(
            {
                "id": "gemini_id",
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": "test prompt",
                "parameters": 1,
            }
        )

    # test error if no prompt is provided
    with pytest.raises(KeyError):
        await gemini_api._obtain_model_inputs(
            {
                "id": "gemini_id",
                "api": "gemini",
                "model_name": "gemini_model_name",
            }
        )

    # test error if no model_name is provided
    with pytest.raises(KeyError):
        await gemini_api._obtain_model_inputs(
            {
                "id": "gemini_id",
                "api": "gemini",
                "prompt": "test prompt",
            }
        )


@pytest.mark.asyncio
async def test_gemini_obtain_model_inputs_safety_filters(
    temporary_data_folders, monkeypatch
):
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    valid_safety_filter_choices = ["none", "few", "default", "some", "most"]

    for safety_filter in valid_safety_filter_choices:
        test_case = await gemini_api._obtain_model_inputs(
            {
                "id": "gemini_id",
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": "test prompt",
                "parameters": {"temperature": 1, "max_output_tokens": 100},
                "safety_filter": safety_filter,
            }
        )
        assert isinstance(test_case, tuple)
        assert len(test_case) == 6
        assert test_case[0] == "test prompt"
        assert test_case[1] == "gemini_model_name"
        assert isinstance(test_case[2], GenerativeModel)
        assert test_case[2]._model_name == "models/gemini_model_name"
        assert test_case[2]._system_instruction is None
        assert isinstance(test_case[3], dict)
        assert test_case[4] == {"temperature": 1, "max_output_tokens": 100}
        assert test_case[5] is None

    # test error if safety filter is not recognised
    with pytest.raises(
        ValueError,
        match=(
            "safety_filter 'unknown-safety-filter' not recognised. Must be one of: "
            "none', 'few', 'default'/'some', 'most'"
        ),
    ):
        await gemini_api._obtain_model_inputs(
            {
                "id": "gemini_id",
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": "test prompt",
                "parameters": {"temperature": 1, "max_output_tokens": 100},
                "safety_filter": "unknown-safety-filter",
            }
        )


@pytest.mark.asyncio
@patch(
    "prompto.apis.gemini.gemini.GeminiAPI._query_string",
    new_callable=AsyncMock,
)
async def test_gemini_query_string(
    mock_query_string, prompt_dict_string, temporary_data_folders, monkeypatch
):
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock the _query_string method to return a response
    mock_query_string.return_value = {**prompt_dict_string, "response": "response text"}

    prompt_dict = await gemini_api.query(prompt_dict_string)

    assert prompt_dict == mock_query_string.return_value
    assert prompt_dict["response"] == "response text"

    mock_query_string.assert_called_once_with(
        prompt_dict=prompt_dict_string, index="NA"
    )


@pytest.mark.asyncio
@patch("prompto.apis.gemini.gemini.GeminiAPI._query_chat", new_callable=AsyncMock)
async def test_gemini_query_chat(
    mock_query_chat, prompt_dict_chat, temporary_data_folders, monkeypatch
):
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock the _query_string method to return a response
    mock_query_chat.return_value = {**prompt_dict_chat, "response": "response text"}

    prompt_dict = await gemini_api.query(prompt_dict_chat)

    assert prompt_dict == mock_query_chat.return_value
    assert prompt_dict["response"] == "response text"

    mock_query_chat.assert_called_once_with(prompt_dict=prompt_dict_chat, index="NA")


@pytest.mark.asyncio
@patch(
    "prompto.apis.gemini.gemini.GeminiAPI._query_history",
    new_callable=AsyncMock,
)
async def test_gemini_query_history(
    mock_query_history, prompt_dict_history, temporary_data_folders, monkeypatch
):
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock the _query_string method to return a response
    mock_query_history.return_value = {
        **prompt_dict_history,
        "response": "response text",
    }

    prompt_dict = await gemini_api.query(prompt_dict_history)

    assert prompt_dict == mock_query_history.return_value
    assert prompt_dict["response"] == "response text"

    mock_query_history.assert_called_once_with(
        prompt_dict=prompt_dict_history, index="NA"
    )


@pytest.mark.asyncio
@patch(
    "prompto.apis.gemini.gemini.GeminiAPI._query_history",
    new_callable=AsyncMock,
)
async def test_gemini_query_history_no_system(
    mock_query_history,
    prompt_dict_history_no_system,
    temporary_data_folders,
    monkeypatch,
):
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # mock the _query_string method to return a response
    mock_query_history.return_value = {
        **prompt_dict_history_no_system,
        "response": "response text",
    }

    prompt_dict = await gemini_api.query(prompt_dict_history_no_system)

    assert prompt_dict == mock_query_history.return_value
    assert prompt_dict["response"] == "response text"

    mock_query_history.assert_called_once_with(
        prompt_dict=prompt_dict_history_no_system, index="NA"
    )


@pytest.mark.asyncio
async def test_gemini_query_error(temporary_data_folders, monkeypatch):
    settings = Settings(data_folder="data")
    log_file = "log.txt"
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY")
    gemini_api = GeminiAPI(settings=settings, log_file=log_file)

    # error if prompt_dict["prompt"] is not of the correct type
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        await gemini_api.query(
            {"api": "gemini", "model_name": "gemini_model_name", "prompt": 1}
        )

    # error if prompt_dict["prompt"] is a list of strings but
    # also contains an incorrect type
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        await gemini_api.query(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": ["prompt 1", "prompt 2", 1],
            }
        )

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # also contains an incorrect type
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        await gemini_api.query(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": [
                    {"role": "system", "parts": "system prompt"},
                    {"role": "user", "parts": "user message"},
                    1,
                ],
            }
        )

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # one of the roles is incorrect
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        await gemini_api.query(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": [
                    {"role": "system", "parts": "system prompt"},
                    {"role": "user", "parts": "user message"},
                    {"role": "incorrect", "parts": "some message"},
                ],
            }
        )

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # the system message is not the first message in the list
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        await gemini_api.query(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": [
                    {"role": "user", "parts": "user message"},
                    {"role": "system", "parts": "system prompt"},
                ],
            }
        )

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # there are multiple system messages in the list
    with pytest.raises(TypeError, match=re.escape(TYPE_ERROR_MSG)):
        await gemini_api.query(
            {
                "api": "gemini",
                "model_name": "gemini_model_name",
                "prompt": [
                    {"role": "system", "parts": "system prompt 1"},
                    {"role": "system", "parts": "system prompt 2"},
                    {"role": "user", "parts": "user message"},
                ],
            }
        )
