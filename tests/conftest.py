import os
import time
from pathlib import Path

import pytest


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

    # create a file with larger number of prompts with different APIs, models (no groups)
    time.sleep(0.01)
    with open(
        Path(tmp_path / "experiment_pipeline" / "input" / "larger_no_groups.jsonl"), "w"
    ) as f:
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
    time.sleep(0.01)
    with open(
        Path(tmp_path / "experiment_pipeline" / "input" / "larger_with_groups.jsonl"),
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

    yield data_dir, dummy_data_dir, utils_dir, experiment_pipeline

    # change back to original directory
    os.chdir(cwd)
