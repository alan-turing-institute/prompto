import pytest
import regex as re

from prompto.apis.anthropic import AnthropicAPI
from prompto.settings import Settings

pytest_plugins = ("pytest_asyncio",)


def test_anthropic_api_init(temporary_data_folders):
    # raise error if no arguments are provided
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        AnthropicAPI()

    # raise error if no settings object is provided
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'settings'"
    ):
        AnthropicAPI(log_file="log.txt")

    # raise error if no log file name is provided
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'log_file'"
    ):
        AnthropicAPI(settings=Settings(data_folder="data"))

    # create a settings object and log file name
    settings = Settings(data_folder="data")
    log_file = "log.txt"

    # intialise the AnthropicAPI class
    anthropic_api = AnthropicAPI(settings=settings, log_file=log_file)

    assert anthropic_api.settings == settings
    assert anthropic_api.log_file == log_file
    assert anthropic_api.api_type == "anthropic"


def test_anthropic_check_environment_variables(temporary_data_folders, monkeypatch):
    # only warn on the ANTHROPIC_API_KEY environment variable
    test_case = AnthropicAPI.check_environment_variables()
    assert len(test_case) == 1
    with pytest.raises(
        Warning, match="Environment variable 'ANTHROPIC_API_KEY' is not set"
    ):
        raise test_case[0]

    # set the ANTHROPIC_API_KEY environment variable
    monkeypatch.setenv("ANTHROPIC_API_KEY", "DUMMY")
    # no warnings or errors
    assert AnthropicAPI.check_environment_variables() == []


def test_anthropic_check_prompt_dict(temporary_data_folders, monkeypatch):
    type_error_msg = (
        "if api == 'anthropic', then the prompt must be a str, list[str], or "
        "list[dict[str,str]] where the dictionary contains the keys 'role' and "
        "'content' only, and the values for 'role' must be one of 'system', 'user' or "
        "'assistant'"
    )

    env_variable_error_msg = (
        "At least one of the environment variables '['ANTHROPIC_API_KEY_anthropic_model_name', "
        "'ANTHROPIC_API_KEY']' must be set"
    )

    # error if prompt_dict["prompt"] is of the correct type
    # also error for no environment variables set
    test_case = AnthropicAPI.check_prompt_dict(
        {"api": "anthropic", "model_name": "anthropic_model_name", "prompt": 1}
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(type_error_msg)):
        raise test_case[0]
    with pytest.raises(KeyError, match=re.escape(env_variable_error_msg)):
        raise test_case[1]

    # error if prompt_dict["prompt"] is a list of strings but
    # also contains an incorrect type
    test_case = AnthropicAPI.check_prompt_dict(
        {
            "api": "anthropic",
            "model_name": "anthropic_model_name",
            "prompt": ["prompt 1", "prompt 2", 1],
        }
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(type_error_msg)):
        raise test_case[0]

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # also contains an incorrect type
    test_case = AnthropicAPI.check_prompt_dict(
        {
            "api": "anthropic",
            "model_name": "anthropic_model_name",
            "prompt": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user message"},
                1,
            ],
        }
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(type_error_msg)):
        raise test_case[0]

    # error if prompt_dict["prompt"] is a list of dictionaries but
    # one of the roles is incorrect
    test_case = AnthropicAPI.check_prompt_dict(
        {
            "api": "anthropic",
            "model_name": "anthropic_model_name",
            "prompt": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user message"},
                {"role": "incorrect", "content": "some message"},
            ],
        }
    )
    assert len(test_case) == 2
    with pytest.raises(TypeError, match=re.escape(type_error_msg)):
        raise test_case[0]

    # error if neither environment variable is set
    test_case = AnthropicAPI.check_prompt_dict(
        {
            "api": "anthropic",
            "model_name": "anthropic_model_name",
            "prompt": "test prompt",
        }
    )
    assert len(test_case) == 1
    with pytest.raises(KeyError, match=re.escape(env_variable_error_msg)):
        raise test_case[0]

    # set the ANTHROPIC_API_KEY environment variable
    monkeypatch.setenv("ANTHROPIC_API_KEY", "DUMMY")
    # error if the model-specific environment variable is not set
    test_case = AnthropicAPI.check_prompt_dict(
        {
            "api": "anthropic",
            "model_name": "anthropic_model_name",
            "prompt": "test prompt",
        }
    )
    assert len(test_case) == 1
    with pytest.raises(
        Warning,
        match=re.escape(
            "Environment variable 'ANTHROPIC_API_KEY_anthropic_model_name' is not set"
        ),
    ):
        raise test_case[0]

    # unset the ANTHROPIC_API_KEY environment variable and
    # set the model-specific environment variable
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY_anthropic_model_name", "DUMMY")
    test_case = AnthropicAPI.check_prompt_dict(
        {
            "api": "anthropic",
            "model_name": "anthropic_model_name",
            "prompt": "test prompt",
        }
    )
    assert len(test_case) == 1
    with pytest.raises(
        Warning, match=re.escape("Environment variable 'ANTHROPIC_API_KEY' is not set")
    ):
        raise test_case[0]

    # full passes
    # set both environment variables
    monkeypatch.setenv("ANTHROPIC_API_KEY", "DUMMY")
    assert (
        AnthropicAPI.check_prompt_dict(
            {
                "api": "anthropic",
                "model_name": "anthropic_model_name",
                "prompt": "test prompt",
            }
        )
        == []
    )
    assert (
        AnthropicAPI.check_prompt_dict(
            {
                "api": "anthropic",
                "model_name": "anthropic_model_name",
                "prompt": ["prompt 1", "prompt 2"],
            }
        )
        == []
    )
    assert (
        AnthropicAPI.check_prompt_dict(
            {
                "api": "anthropic",
                "model_name": "anthropic_model_name",
                "prompt": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "user message 1"},
                    {"role": "assistant", "content": "assistant message"},
                    {"role": "user", "content": "user message 2"},
                ],
            }
        )
        == []
    )
