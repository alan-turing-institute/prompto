import logging
import os

import pytest

from prompto.experiment import Experiment
from prompto.settings import Settings

pytest_plugins = ("pytest_asyncio",)


def example_evaluation_func1(prompt_dict):
    """
    Example evaluation function that checks if the word "test" is in the response.
    """
    prompt_dict["evaluation"] = "test" in prompt_dict["response"]
    return prompt_dict


def example_evaluation_func2(prompt_dict):
    """
    Example evaluation function that returns the length of the response.
    """
    prompt_dict["length_of_response"] = len(prompt_dict["response"])
    return prompt_dict


@pytest.mark.asyncio
async def test_generate_text_success_no_index(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs using "test" api
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
    assert result["response"] == "This is a test response"

    # check logs for success message
    log_msg = (
        "Response received for model test (i=NA, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_success_no_index_with_id(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs using "test" api
    result = await experiment.generate_text(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=None,
    )

    assert result["api"] == "test"
    assert result["prompt"] == "test prompt"
    assert result["response"] == "This is a test response"

    # check logs for success message
    log_msg = (
        "Response received for model test (i=NA, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_success_with_index_str(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs using "test" api
    result = await experiment.generate_text(
        prompt_dict={
            "api": "test",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index="index_test",
    )

    assert result["api"] == "test"
    assert result["prompt"] == "test prompt"
    assert result["response"] == "This is a test response"

    # check logs for success message
    log_msg = (
        "Response received for model test (i=index_test, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_success_with_index_str_with_id(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs using "test" api
    result = await experiment.generate_text(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index="index_test",
    )

    assert result["api"] == "test"
    assert result["prompt"] == "test prompt"
    assert result["response"] == "This is a test response"

    # check logs for success message
    log_msg = (
        "Response received for model test (i=index_test, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_success_with_index_int(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs using "test" api
    result = await experiment.generate_text(
        prompt_dict={
            "api": "test",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
    )

    assert result["api"] == "test"
    assert result["prompt"] == "test prompt"
    assert "response" in result

    # check logs for success message
    log_msg = (
        "Response received for model test (i=2, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_success_with_index_int_with_id(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs using "test" api
    result = await experiment.generate_text(
        prompt_dict={
            "id": "test_id",
            "api": "test",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=2,
    )

    assert result["api"] == "test"
    assert result["prompt"] == "test prompt"
    assert "response" in result

    # check logs for success message
    log_msg = (
        "Response received for model test (i=2, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_error(temporary_data_folder_for_processing, caplog):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # check raises error from "test" model when we request one
    with pytest.raises(ValueError):
        result = await experiment.generate_text(
            prompt_dict={
                "api": "test",
                "prompt": "test prompt",
                "parameters": {"raise_error": "True"},
            },
            index="some_index",
        )

    # check logs for success message
    log_msg = (
        "Error with model test (i=some_index, id=NA)\n"
        "Prompt: test prompt...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_error_with_id(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # check raises error from "test" model when we request one
    with pytest.raises(ValueError):
        result = await experiment.generate_text(
            prompt_dict={
                "id": "test_id",
                "api": "test",
                "prompt": "test prompt",
                "parameters": {"raise_error": "True"},
            },
            index="some_index",
        )

    # check logs for success message
    log_msg = (
        "Error with model test (i=some_index, id=test_id)\n"
        "Prompt: test prompt...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_not_implemented_error(
    temporary_data_folder_for_processing,
):
    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # check raises error if api is not found
    with pytest.raises(
        NotImplementedError,
        match="API api-that-does-not-exist not recognised or implemented",
    ):
        await experiment.generate_text(
            prompt_dict={
                "api": "api-that-does-not-exist",
                "prompt": "test prompt",
            },
            index=None,
        )


@pytest.mark.asyncio
async def test_generate_text_error_api_not_specified(
    temporary_data_folder_for_processing,
):
    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # check raises error if api is not found
    with pytest.raises(
        KeyError, match="API is not specified in the prompt_dict. Must have 'api' key"
    ):
        await experiment.generate_text(
            prompt_dict={
                "prompt": "test prompt",
            },
            index=None,
        )


@pytest.mark.asyncio
async def test_generate_text_with_1evaluation(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs using "test" api
    result = await experiment.generate_text(
        prompt_dict={
            "api": "test",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=None,
        evaluation_funcs=[example_evaluation_func1],
    )

    # Normal repsonses should remain unchanged
    assert result["api"] == "test"
    assert result["prompt"] == "test prompt"
    assert result["response"] == "This is a test response"

    # Check evaluation func has been successfully applied
    assert result["evaluation"] is True

    # check logs for success message
    log_msg = (
        "Response received for model test (i=NA, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_generate_text_with_2evaluations(
    temporary_data_folder_for_processing, caplog
):
    caplog.set_level(logging.INFO)

    # create a settings object
    settings = Settings(data_folder="data")

    # create an experiment object
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    # await generate_text method on different inputs using "test" api
    result = await experiment.generate_text(
        prompt_dict={
            "api": "test",
            "prompt": "test prompt",
            "parameters": {"raise_error": "False"},
        },
        index=None,
        evaluation_funcs=[example_evaluation_func1, example_evaluation_func2],
    )

    # Normal repsonses should remain unchanged
    assert result["api"] == "test"
    assert result["prompt"] == "test prompt"
    assert result["response"] == "This is a test response"

    # Check evaluation func has been successfully applied
    assert result["evaluation"] is True
    assert result["length_of_response"] == 21

    # check logs for success message
    log_msg = (
        "Response received for model test (i=NA, id=NA)\n"
        "Prompt: test prompt...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
