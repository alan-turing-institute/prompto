import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import AsyncAnthropic

from prompto.apis.anthropic import AnthropicAPI
from prompto.settings import Settings

pytest_plugins = ("pytest_asyncio",)

PROMPT_DICT_HISTORY = {
    "id": "anthropic_1",
    "api": "anthropic",
    "model_name": "anthropic_model_name",
    "prompt": [
        {"role": "system", "content": "test system prompt"},
        {"role": "user", "content": "user message"},
    ],
    "parameters": {"temperature": 1, "max_tokens": 100},
}
