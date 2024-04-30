import pytest

from batch_llm.experiment_processing import Experiment
from batch_llm.settings import Settings


def test_experiment_init_errors(temporary_data_folders):
    # not passing in file_name or settings should raise TypeError as they're required
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        Experiment()

    # passing in file_name and no settings should raise TypeError as settings is required
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        Experiment("test.jsonl")

    # passing in settings and no file_name should raise TypeError as file_name is required
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        Experiment(settings=Settings())

    # passing in a filename that is not a .jsonl file should raise a ValueError
    with pytest.raises(ValueError, match="Experiment file must be a jsonl file"):
        Experiment("test.txt", settings=Settings())

    # passing in a filename that is not in settings.input_folder should raise a FileNotFoundError
    with pytest.raises(
        FileNotFoundError,
        match="Experiment file 'data/input/test.jsonl' does not exist",
    ):
        Experiment("test.jsonl", settings=Settings())


def test_experiment_init(temporary_data_folders):
    # create a settings object
    settings = Settings(data_folder="data", max_queries=50, max_attempts=5)

    # create a jsonl file in the input folder (which is created when initialising Settings object)
    with open("data/input/test_in_input.jsonl", "w") as f:
        f.write('{"id": 0, "prompt": "test prompt 0", "api": "test"}\n')
        f.write('{"id": 1, "prompt": "test prompt 1", "api": "test"}\n')

    # create an experiment object
    experiment = Experiment("test_in_input.jsonl", settings=settings)

    # check the experiment object has the correct attributes
    assert experiment.file_name == "test_in_input.jsonl"
    assert experiment.experiment_name == "test_in_input"
    assert experiment.settings == settings
    assert experiment.output_folder == "data/output/test_in_input"
    assert experiment.input_file_path == "data/input/test_in_input.jsonl"
    assert (
        experiment.output_completed_file_path
        == f"data/output/test_in_input/{experiment.creation_time}-completed-test_in_input.jsonl"
    )
    assert (
        experiment.output_input_file_out_path
        == f"data/output/test_in_input/{experiment.creation_time}-input-test_in_input.jsonl"
    )
    assert experiment.experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test"},
        {"id": 1, "prompt": "test prompt 1", "api": "test"},
    ]
    assert experiment.number_queries == 2
    assert isinstance(experiment.creation_time, str)
    assert (
        experiment.log_file
        == f"data/output/test_in_input/{experiment.creation_time}-log.txt"
    )

    # test str method
    assert str(experiment) == "test_in_input.jsonl"
