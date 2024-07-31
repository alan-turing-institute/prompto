import os
from unittest.mock import AsyncMock, patch

import pytest

from prompto.apis.anthropic import AnthropicAPI
from prompto.settings import Settings

os.environ["ANTHROPIC_API_KEY"] = "DUMMY"

# Set default settings
settings = Settings()

PROMPT_DICT = {
    "id": 0,
    "api": "anthropic",
    "model_name": "claude-3-haiku-20240307",
    "prompt": "How does technology impact us?",
    "parameters": {"temperature": 1, "max_tokens": 100},
}

LOG_FILE = "log.txt"


class MockResponse:
    def __init__(self, data):
        self.data = data

    async def json(self):
        return self.data


@pytest.mark.asyncio
async def test_query_string():
    # Intialize the AnthropicAPI class
    anthropic_api = AnthropicAPI(settings=settings, log_file=LOG_FILE)

    with patch(
        "prompto.apis.anthropic.anthropic.AsyncAnthropic", new_callable=AsyncMock
    ) as mock_anthropic:

        # This fails because the mock response needs to be a "Messages" object not a json.
        mock_anthropic.return_value = {"prompt": "42"}

        # Query the string
        prompt_dict = await anthropic_api._query_string(PROMPT_DICT, index=0)

    assert prompt_dict["prompt"] == "42"
