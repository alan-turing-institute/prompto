import os
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


class CopyingAsyncMock(AsyncMock):
    # used in situations where mocking mutable arguments
    def __call__(self, /, *args, **kwargs):
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        return super().__call__(*args, **kwargs)


@pytest.fixture(autouse=True)
def change_test_dir(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """
    Change to the working directory to that of the test file.
    """
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def temporary_data_folders(tmp_path: Path):
    """
    Creates a temporary folder structure for testing.

    Has the following structure:
    tmp_path
    ├── data/
    ├── dummy_data/
    ├── utils/
        └── first.jsonl
        └── second.jsonl
        └── third.jsonl
    ├── experiment_pipeline/
        ├── input/
            └── first.jsonl
            └── second.jsonl
        ├── output/
        ├── media/
    ├── test.txt
    └── test.jsonl
    """
    # create data folders
    data_dir = Path(tmp_path / "data").mkdir()
    dummy_data_dir = Path(tmp_path / "dummy_data").mkdir()

    # create a txt file in the folder
    with open(Path(tmp_path / "test.txt"), "w") as f:
        f.write("test line")

    # create a jsonl file in the folder
    with open(Path(tmp_path / "test.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt", "api": "test"}\n')

    # create utils folder which we use to test the sorting of files
    utils_dir = Path(tmp_path / "utils").mkdir()
    with open(Path(tmp_path / "utils" / "first.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 1", "api": "test"}\n')

    time.sleep(0.01)
    with open(Path(tmp_path / "utils" / "second.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 2", "api": "test"}\n')

    time.sleep(0.01)
    with open(Path(tmp_path / "utils" / "third.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 3", "api": "test"}\n')

    # create a folder for testing the experiment pipeline
    experiment_pipeline = Path(tmp_path / "experiment_pipeline").mkdir()
    # create subfolders for the experiment pipeline
    Path(tmp_path / "experiment_pipeline" / "input").mkdir()
    Path(tmp_path / "experiment_pipeline" / "output").mkdir()
    Path(tmp_path / "experiment_pipeline" / "media").mkdir()
    # add some dummy jsonl files to the input folder
    with open(
        Path(tmp_path / "experiment_pipeline" / "input" / "first.jsonl"), "w"
    ) as f:
        f.write('{"prompt": "test prompt 1", "api": "test"}\n')
        f.write('{"prompt": "test prompt 2", "api": "test"}\n')
        f.write('{"prompt": "test prompt 3", "api": "test"}\n')

    time.sleep(0.01)
    with open(
        Path(tmp_path / "experiment_pipeline" / "input" / "second.jsonl"), "w"
    ) as f:
        f.write('{"prompt": "test prompt 2", "api": "test"}\n')

    # store current working directory
    cwd = os.getcwd()

    # change to temporary directory
    os.chdir(tmp_path)

    yield data_dir, dummy_data_dir, utils_dir, experiment_pipeline

    # change back to original directory
    os.chdir(cwd)


@pytest.fixture()
def temporary_data_folder_for_grouping_prompts(tmp_path: Path):
    """
    Creates a temporary folder structure for testing grouping prompts.

    Has the following structure:
    tmp_path
    ├── data/
        ├── input/
            └── larger_no_groups.jsonl
            └── larger_with_groups.jsonl
        ├── output/
        ├── media/
    """
    # create a folder for testing the experiment pipeline
    data_dir = Path(tmp_path / "data").mkdir()
    # create subfolders for the experiment pipeline
    Path(tmp_path / "data" / "input").mkdir()
    Path(tmp_path / "data" / "output").mkdir()
    Path(tmp_path / "data" / "media").mkdir()

    # create a file with larger number of prompts with different APIs, models (no groups)
    with open(Path(tmp_path / "data" / "input" / "larger_no_groups.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 1", "api": "test", "model_name": "model1"}\n')
        f.write('{"prompt": "test prompt 2", "api": "test"}\n')
        f.write('{"prompt": "test prompt 3", "api": "test", "model_name": "model1"}\n')
        f.write('{"prompt": "test prompt 4", "api": "test", "model_name": "model3"}\n')
        f.write('{"prompt": "test prompt 5", "api": "test", "model_name": "model2"}\n')
        f.write('{"prompt": "test prompt 6", "api": "test", "model_name": "model3"}\n')
        f.write('{"prompt": "test prompt 7", "api": "test", "model_name": "model3"}\n')
        f.write('{"prompt": "test prompt 8", "api": "test"}\n')
        f.write(
            '{"prompt": "gemini prompt 1", "api": "gemini", "model_name": "gemini-pro"}\n'
        )
        f.write(
            '{"prompt": "gemini prompt 2", "api": "gemini", "model_name": "gemini-pro"}\n'
        )
        f.write('{"prompt": "gemini prompt 3", "api": "gemini"}\n')
        f.write(
            '{"prompt": "gemini prompt 4", "api": "gemini", "model_name": "gemini-pro"}\n'
        )
        f.write('{"prompt": "gemini prompt 5", "api": "gemini"}\n')
        f.write(
            '{"prompt": "azure-openai prompt 1", "api": "azure-openai", "model_name": "gpt3"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 2", "api": "azure-openai", "model_name": "gpt4"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 3", "api": "azure-openai", "model_name": "gpt3"}\n'
        )
        f.write('{"prompt": "azure-openai prompt 4", "api": "azure-openai"}\n')
        f.write(
            '{"prompt": "azure-openai prompt 5", "api": "azure-openai", "model_name": "gpt3"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 6", "api": "azure-openai", "model_name": "gpt4"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 7", "api": "azure-openai", "model_name": "gpt3.5"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 8", "api": "azure-openai", "model_name": "gpt3.5"}\n'
        )

    # create a file with larger number of prompts with different APIs, models and groups
    with open(
        Path(tmp_path / "data" / "input" / "larger_with_groups.jsonl"),
        "w",
    ) as f:
        f.write('{"prompt": "test prompt 1", "api": "test", "model_name": "model1"}\n')
        f.write('{"prompt": "test prompt 2", "api": "test"}\n')
        f.write(
            '{"prompt": "test prompt 3", "api": "test", "model_name": "model1", "group": "group1"}\n'
        )
        f.write('{"prompt": "test prompt 4", "api": "test", "model_name": "model3"}\n')
        f.write(
            '{"prompt": "test prompt 5", "api": "test", "model_name": "model2", "group": "group1"}\n'
        )
        f.write(
            '{"prompt": "test prompt 6", "api": "test", "model_name": "model3", "group": "group1"}\n'
        )
        f.write(
            '{"prompt": "test prompt 7", "api": "test", "model_name": "model3", "group": "group2"}\n'
        )
        f.write('{"prompt": "test prompt 8", "api": "test", "group": "group2"}\n')
        f.write(
            '{"prompt": "gemini prompt 1", "api": "gemini", "model_name": "gemini-pro", "group": "group1"}\n'
        )
        f.write(
            '{"prompt": "gemini prompt 2", "api": "gemini", "model_name": "gemini-pro", "group": "group2"}\n'
        )
        f.write('{"prompt": "gemini prompt 3", "api": "gemini", "group": "group1"}\n')
        f.write(
            '{"prompt": "gemini prompt 4", "api": "gemini", "model_name": "gemini-pro"}\n'
        )
        f.write('{"prompt": "gemini prompt 5", "api": "gemini"}\n')
        f.write(
            '{"prompt": "azure-openai prompt 1", "api": "azure-openai", "model_name": "gpt3", "group": "group1"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 2", "api": "azure-openai", "model_name": "gpt4"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 3", "api": "azure-openai", "model_name": "gpt3", "group": "group1"}\n'
        )
        f.write('{"prompt": "azure-openai prompt 4", "api": "azure-openai"}\n')
        f.write(
            '{"prompt": "azure-openai prompt 5", "api": "azure-openai", "model_name": "gpt3"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 6", "api": "azure-openai", "model_name": "gpt4", "group": "group1"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 7", "api": "azure-openai", "model_name": "gpt3.5"}\n'
        )
        f.write(
            '{"prompt": "azure-openai prompt 8", "api": "azure-openai", "model_name": "gpt3.5", "group": "group2"}\n'
        )

    # store current working directory
    cwd = os.getcwd()

    # change to temporary directory
    os.chdir(tmp_path)

    yield data_dir

    # change back to original directory
    os.chdir(cwd)


@pytest.fixture()
def temporary_rate_limit_doc_examples(tmp_path: Path):
    """
    Creates a temporary folder structure for testing rate limit examples
    from the docs/rate_limit.md file.

    Has the following structure:
    tmp_path
    ├── data/
        ├── input/
            └── rate_limit_docs_example.jsonl
            └── rate_limit_docs_example_groups.jsonl
            └── rate_limit_docs_example_groups_2.jsonl
        ├── output/
        ├── media/
    """
    # create a folder for testing the experiment pipeline
    data_dir = Path(tmp_path / "data").mkdir()
    # create subfolders for the experiment pipeline
    Path(tmp_path / "data" / "input").mkdir()
    Path(tmp_path / "data" / "output").mkdir()
    Path(tmp_path / "data" / "media").mkdir()

    # add a jsonl file with rate limit examples from the docs/rate_limit.md fileå
    with open(
        Path(tmp_path / "data" / "input" / "rate_limit_docs_example.jsonl"),
        "w",
    ) as f:
        f.write(
            '{"id": 0, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 1, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 2, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 3, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 4, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 5, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 6, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 7, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 8, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 9, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 10, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 11, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of Germany?"}\n'
        )

    # add a jsonl file with rate limit examples from the docs/rate_limit.md file
    with open(
        Path(tmp_path / "data" / "input" / "rate_limit_docs_example_groups.jsonl"),
        "w",
    ) as f:
        f.write(
            '{"id": 0, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of France?", "group": "group1"}\n'
        )
        f.write(
            '{"id": 1, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of Germany?", "group": "group2"}\n'
        )
        f.write(
            '{"id": 2, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of France?", "group": "group1"}\n'
        )
        f.write(
            '{"id": 3, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of Germany?", "group": "group2"}\n'
        )
        f.write(
            '{"id": 4, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of France?", "group": "group1"}\n'
        )
        f.write(
            '{"id": 5, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of Germany?", "group": "group2"}\n'
        )
        f.write(
            '{"id": 6, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of France?", "group": "group1"}\n'
        )
        f.write(
            '{"id": 7, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of Germany?", "group": "group2"}\n'
        )
        f.write(
            '{"id": 8, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of France?", "group": "group3"}\n'
        )
        f.write(
            '{"id": 9, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of Germany?", "group": "group3"}\n'
        )
        f.write(
            '{"id": 10, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of France?", "group": "group3"}\n'
        )
        f.write(
            '{"id": 11, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of Germany?", "group": "group3"}\n'
        )

    # add a jsonl file with rate limit examples from the docs/rate_limit.md file
    with open(
        Path(tmp_path / "data" / "input" / "rate_limit_docs_example_groups_2.jsonl"),
        "w",
    ) as f:
        f.write(
            '{"id": 0, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 1, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 2, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 3, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 4, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 5, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 6, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of France?"}\n'
        )
        f.write(
            '{"id": 7, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of Germany?"}\n'
        )
        f.write(
            '{"id": 8, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of France?", "group": "group1"}\n'
        )
        f.write(
            '{"id": 9, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of Germany?", "group": "group1"}\n'
        )
        f.write(
            '{"id": 10, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of France?", "group": "group1"}\n'
        )
        f.write(
            '{"id": 11, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of Germany?", "group": "group1"}\n'
        )
        f.write(
            '{"id": 12, "api": "ollama", "model_name": "gemma", "prompt": "What is the capital of France?", "group": "group2"}\n'
        )
        f.write(
            '{"id": 13, "api": "ollama", "model_name": "gemma", "prompt": "What is the capital of Germany?", "group": "group2"}\n'
        )
        f.write(
            '{"id": 14, "api": "ollama", "model_name": "phi3", "prompt": "What is the capital of France?", "group": "group2"}\n'
        )
        f.write(
            '{"id": 15, "api": "ollama", "model_name": "phi3", "prompt": "What is the capital of Germany?", "group": "group2"}\n'
        )

    # store current working directory
    cwd = os.getcwd()

    # change to temporary directory
    os.chdir(tmp_path)

    yield data_dir

    # change back to original directory
    os.chdir(cwd)


@pytest.fixture()
def temporary_data_folder_for_processing(tmp_path: Path):
    """
    Creates a temporary folder structure for testing processing of experiments.

    Has the following structure:
    tmp_path
    ├── data/
        ├── input/
            └── test_experiment.jsonl
        ├── output/
        ├── media/
    """
    # create a folder for testing the experiment pipeline
    data_dir = Path(tmp_path / "data").mkdir()
    # create subfolders for the experiment pipeline
    Path(tmp_path / "data" / "input").mkdir()
    Path(tmp_path / "data" / "output").mkdir()
    Path(tmp_path / "data" / "media").mkdir()

    # create a file with larger number of prompts with different APIs, models (no groups)
    with open(Path(tmp_path / "data" / "input" / "test_experiment.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 1", "api": "test", "model_name": "model1"}\n')
        f.write('{"prompt": "test prompt 2", "api": "test"}\n')
        f.write('{"prompt": "test prompt 3", "api": "test", "model_name": "model1"}\n')
        f.write('{"prompt": "test prompt 4", "api": "test", "model_name": "model3"}\n')
        f.write('{"prompt": "test prompt 5", "api": "test", "model_name": "model2"}\n')
        f.write('{"prompt": "test prompt 6", "api": "test", "model_name": "model3"}\n')
        f.write('{"prompt": "test prompt 7", "api": "test", "model_name": "model3"}\n')
        f.write('{"prompt": "test prompt 8", "api": "test"}\n')

    # store current working directory
    cwd = os.getcwd()

    # change to temporary directory
    os.chdir(tmp_path)

    yield data_dir

    # change back to original directory
    os.chdir(cwd)
