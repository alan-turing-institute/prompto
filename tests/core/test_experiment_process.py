import logging
import os

import pytest

from prompto.experiment import Experiment
from prompto.settings import Settings

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_generate_text(temporary_data_folder_for_processing, caplog):

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs
    result = await experiment.generate_text(
        prompt_dict={
            "api": "test",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=None,
    )

    assert result["api"] == "test"
    assert result["prompt"] == "test prompt"
    assert "response" in result
