import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import AsyncAnthropic

from prompto.apis.anthropic import AnthropicAPI
from prompto.settings import Settings

pytest_plugins = ("pytest_asyncio",)

PROMPT_DICT_CHAT = {
    "id": "anthropic_id",
    "api": "anthropic",
    "model_name": "anthropic_model_name",
    "prompt": ["test chat 1", "test chat 2"],
    "parameters": {"temperature": 1, "max_tokens": 100},
}


@pytest.mark.asyncio
async def test_anthropic_query_chat_no_env_var(temporary_data_folders, caplog):
    caplog.set_level(logging.INFO)

    # create a settings object and log file name
    settings = Settings(data_folder="data")
    log_file = "log.txt"

    # intialise the AnthropicAPI class
    anthropic_api = AnthropicAPI(settings=settings, log_file=log_file)

    # raise error if no environment variable is set
    with pytest.raises(
        KeyError,
        match=(
            "Neither 'ANTHROPIC_API_KEY' nor 'ANTHROPIC_API_KEY_anthropic_model_name' "
            "environment variable is set."
        ),
    ):
        await anthropic_api._query_chat(PROMPT_DICT_CHAT, index=0)
