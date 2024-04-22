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
def temporary_utils_folder(tmp_path: Path):
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
    # create a txt file in the folder
    with open(Path(tmp_path / "test.txt"), "w") as f:
        f.write("test line")

    # create utils folder which we use to test the sorting of files
    utils_dir = Path(tmp_path / "utils").mkdir()
    with open(Path(tmp_path / "utils" / "first.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 1", "model": "test"}\n')
    time.sleep(0.01)
    with open(Path(tmp_path / "utils" / "second.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 2", "model": "test"}\n')
    time.sleep(0.01)
    with open(Path(tmp_path / "utils" / "third.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 3", "model": "test"}\n')

    # store current working directory
    cwd = os.getcwd()

    # change to temporary directory
    os.chdir(tmp_path)

    yield utils_dir

    # change back to original directory
    os.chdir(cwd)


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
        f.write('{"prompt": "test prompt", "model": "test"}\n')

    # create utils folder which we use to test the sorting of files
    utils_dir = Path(tmp_path / "utils").mkdir()
    with open(Path(tmp_path / "utils" / "first.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 1", "model": "test"}\n')
    time.sleep(0.001)
    with open(Path(tmp_path / "utils" / "second.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 2", "model": "test"}\n')
    time.sleep(0.001)
    with open(Path(tmp_path / "utils" / "third.jsonl"), "w") as f:
        f.write('{"prompt": "test prompt 3", "model": "test"}\n')

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
        f.write('{"prompt": "test prompt 1", "model": "test"}\n')
        f.write('{"prompt": "test prompt 2", "model": "test"}\n')
        f.write('{"prompt": "test prompt 3", "model": "test"}\n')
    time.sleep(0.001)
    with open(
        Path(tmp_path / "experiment_pipeline" / "input" / "second.jsonl"), "w"
    ) as f:
        f.write('{"prompt": "test prompt 2", "model": "test"}\n')

    # store current working directory
    cwd = os.getcwd()

    # change to temporary directory
    os.chdir(tmp_path)

    yield data_dir, dummy_data_dir, utils_dir, experiment_pipeline

    # change back to original directory
    os.chdir(cwd)
