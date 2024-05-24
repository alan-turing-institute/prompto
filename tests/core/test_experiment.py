import pytest

from prompto.experiment_processing import Experiment
from prompto.settings import Settings


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
    assert experiment._experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test"},
        {"id": 1, "prompt": "test prompt 1", "api": "test"},
    ]
    # check property getter for experiment_prompts
    assert experiment.experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test"},
        {"id": 1, "prompt": "test prompt 1", "api": "test"},
    ]
    assert experiment.number_queries == 2
    assert isinstance(experiment.creation_time, str)
    assert (
        experiment.log_file
        == f"data/output/test_in_input/{experiment.creation_time}-test_in_input-log.txt"
    )

    # test str method
    assert str(experiment) == "test_in_input.jsonl"

    # test that grouped experiments have not been created yet
    assert experiment._grouped_experiment_prompts == {}


def test_experiment_grouped_prompts_simple(temporary_data_folders):
    # create a settings object
    settings = Settings(data_folder="data", max_queries=50, max_attempts=5)

    # create a jsonl file in the input folder (which is created when initialising Settings object)
    with open("data/input/test_in_input.jsonl", "w") as f:
        f.write('{"id": 0, "prompt": "test prompt 0", "api": "test"}\n')
        f.write('{"id": 1, "prompt": "test prompt 1", "api": "test"}\n')

    # create an experiment object
    experiment = Experiment("test_in_input.jsonl", settings=settings)

    # check the experiment_prompts attribute is correct
    assert experiment._experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test"},
        {"id": 1, "prompt": "test prompt 1", "api": "test"},
    ]
    # check property getter for experiment_prompts
    assert experiment.experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test"},
        {"id": 1, "prompt": "test prompt 1", "api": "test"},
    ]

    # test that grouped experiments have not been created yet
    assert experiment._grouped_experiment_prompts == {}

    # when the attribute is called, it will get the grouped prompts by calling group_prompts method
    grouped_prompts = experiment.grouped_experiment_prompts
    # when settings.max_queries_dict is empty, we just group by "group" or "api"
    # here, prompts only have "api" key with "test", so should be a dictionary of one "test" key
    assert grouped_prompts == {
        "test": {"prompt_dicts": experiment.experiment_prompts, "rate_limit": 50}
    }
    # check that this attribute comes from the group_prompts method
    assert experiment._grouped_experiment_prompts == experiment.group_prompts()


def test_experiment_experiment_prompts_getter(temporary_data_folders):
    # create a settings object
    settings = Settings(data_folder="experiment_pipeline")

    # create an experiment object
    experiment = Experiment("first.jsonl", settings=settings)

    assert experiment._experiment_prompts == [
        {"prompt": "test prompt 1", "api": "test"},
        {"prompt": "test prompt 2", "api": "test"},
        {"prompt": "test prompt 3", "api": "test"},
    ]
    assert experiment.experiment_prompts == [
        {"prompt": "test prompt 1", "api": "test"},
        {"prompt": "test prompt 2", "api": "test"},
        {"prompt": "test prompt 3", "api": "test"},
    ]

    # set it to a different list
    # (manually circumventing the setter error)
    experiment._experiment_prompts = [{"prompt": "hello", "api": "world"}]
    assert experiment.experiment_prompts == [{"prompt": "hello", "api": "world"}]


def test_experiment_experiment_prompts_setter(temporary_data_folders):
    # create a settings object
    settings = Settings(data_folder="experiment_pipeline")

    # create an experiment object
    experiment = Experiment("first.jsonl", settings=settings)

    # should raise an AttributeError if trying to set the experiment_prompts attribute
    with pytest.raises(
        AttributeError, match="Cannot set the experiment_prompts attribute"
    ):
        experiment.experiment_prompts = [{"prompt": "hello", "api": "world"}]

    # test that the experiment_prompts attribute is still the same after the error
    assert experiment.experiment_prompts == [
        {"prompt": "test prompt 1", "api": "test"},
        {"prompt": "test prompt 2", "api": "test"},
        {"prompt": "test prompt 3", "api": "test"},
    ]


def test_experiment_grouped_prompts_getter(temporary_data_folders):
    # create a settings object
    settings = Settings(data_folder="experiment_pipeline")

    # create an experiment object
    experiment = Experiment("first.jsonl", settings=settings)

    # check only created when it's called (initially it's an empty dict)
    assert experiment._grouped_experiment_prompts == {}
    assert experiment.experiment_prompts == [
        {"prompt": "test prompt 1", "api": "test"},
        {"prompt": "test prompt 2", "api": "test"},
        {"prompt": "test prompt 3", "api": "test"},
    ]
    assert experiment.grouped_experiment_prompts == {
        "test": {"prompt_dicts": experiment.experiment_prompts, "rate_limit": 10}
    }

    # set it to a different dictionary
    # (manually circumventing the setter error)
    experiment._grouped_experiment_prompts = {"hello": "world"}
    assert experiment.grouped_experiment_prompts == {"hello": "world"}


def test_experiment_grouped_prompts_setter(temporary_data_folders):
    # create a settings object
    settings = Settings(data_folder="experiment_pipeline")

    # create an experiment object
    experiment = Experiment("first.jsonl", settings=settings)

    # check it's an empty dict initially
    assert experiment._grouped_experiment_prompts == {}

    # try to set the grouped_experiment_prompts attribute when it's still an empty dict
    with pytest.raises(
        AttributeError, match="Cannot set the grouped_experiment_prompts attribute"
    ):
        experiment.grouped_experiment_prompts = {"hello": "world"}

    # check only created when it's called and it's still an empty dict after error
    assert experiment._grouped_experiment_prompts == {}
    assert experiment.experiment_prompts == [
        {"prompt": "test prompt 1", "api": "test"},
        {"prompt": "test prompt 2", "api": "test"},
        {"prompt": "test prompt 3", "api": "test"},
    ]
    assert experiment.grouped_experiment_prompts == {
        "test": {"prompt_dicts": experiment.experiment_prompts, "rate_limit": 10}
    }

    # try to set the grouped_experiment_prompts attribute when it's been set
    with pytest.raises(
        AttributeError, match="Cannot set the grouped_experiment_prompts attribute"
    ):
        experiment.grouped_experiment_prompts = {"hello": "world"}


def test_experiment_grouped_prompts_default_no_groups(temporary_data_folders):
    # create a settings object without max_queries_dict (i.e. it's {} by default)
    settings = Settings(
        data_folder="experiment_pipeline", max_queries=50, max_attempts=5
    )
    assert settings.max_queries_dict == {}

    # parallel is by default False too but we will group the prompts when calling the attribute
    assert settings.parallel is False

    # create an experiment object
    experiment = Experiment("larger_no_groups.jsonl", settings=settings)

    # since we haven't passed in a max_queries_dict, the grouped prompts will be grouped
    # by the "api" key
    assert experiment.grouped_experiment_prompts == {
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 3", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 5", "api": "test", "model_name": "model2"},
                {"prompt": "test prompt 6", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 7", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 8", "api": "test"},
            ],
            "rate_limit": 50,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 3", "api": "gemini"},
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 50,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 50,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "test": "8 queries at 50 queries per minute",
        "gemini": "5 queries at 50 queries per minute",
        "azure-openai": "8 queries at 50 queries per minute",
    }


def test_experiment_grouped_prompts_apis_only_no_groups(temporary_data_folders):
    # specify rate limits for "test" and "gemini", but leave "azure-openai" to default
    # also add another key which should be ignored but present in the group_experiment_prompts
    max_queries_dict = {"test": 100, "gemini": 200, "ignored": 10}

    # create a settings object
    settings = Settings(
        data_folder="experiment_pipeline",
        max_queries=50,
        max_attempts=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("larger_no_groups.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "ignored": {
            "prompt_dicts": [],
            "rate_limit": 10,
        },
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 3", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 5", "api": "test", "model_name": "model2"},
                {"prompt": "test prompt 6", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 7", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 8", "api": "test"},
            ],
            "rate_limit": 100,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 3", "api": "gemini"},
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 200,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 50,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "ignored": "0 queries at 10 queries per minute",
        "test": "8 queries at 100 queries per minute",
        "gemini": "5 queries at 200 queries per minute",
        "azure-openai": "8 queries at 50 queries per minute",
    }


def test_experiment_grouped_prompts_apis_and_models_no_groups_v1(
    temporary_data_folders,
):
    # specify rate limits for "test" and "gemini" apis and some models,
    # but only specify overall rate limit for "azure-openai"
    # also add other key which should be ignored but
    # present in the group_experiment_prompts
    max_queries_dict = {
        "test": {"model2": 100, "model1": 200},
        "gemini": {"default": 25, "gemini-pro": 20},
        "azure-openai": 10,
        "ignored": 10,
        "ignored-with-models": {"default": 15, "ignored-model": 5},
    }

    # create a settings object
    settings = Settings(
        data_folder="experiment_pipeline",
        max_queries=50,
        max_attempts=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("larger_no_groups.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "ignored": {
            "prompt_dicts": [],
            "rate_limit": 10,
        },
        "ignored-with-models": {
            "prompt_dicts": [],
            "rate_limit": 15,
        },
        "ignored-with-models-ignored-model": {
            "prompt_dicts": [],
            "rate_limit": 5,
        },
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 6", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 7", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 8", "api": "test"},
            ],
            "rate_limit": 50,
        },
        "test-model1": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 3", "api": "test", "model_name": "model1"},
            ],
            "rate_limit": 200,
        },
        "test-model2": {
            "prompt_dicts": [
                {"prompt": "test prompt 5", "api": "test", "model_name": "model2"}
            ],
            "rate_limit": 100,
        },
        "gemini-gemini-pro": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
            ],
            "rate_limit": 20,
        },
        "gemini": {
            "prompt_dicts": [
                {"prompt": "gemini prompt 3", "api": "gemini"},
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 25,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 10,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "ignored": "0 queries at 10 queries per minute",
        "ignored-with-models": "0 queries at 15 queries per minute",
        "ignored-with-models-ignored-model": "0 queries at 5 queries per minute",
        "test": "5 queries at 50 queries per minute",
        "test-model1": "2 queries at 200 queries per minute",
        "test-model2": "1 queries at 100 queries per minute",
        "gemini": "2 queries at 25 queries per minute",
        "gemini-gemini-pro": "3 queries at 20 queries per minute",
        "azure-openai": "8 queries at 10 queries per minute",
    }


def test_experiment_grouped_prompts_apis_and_models_no_groups_v2(
    temporary_data_folders,
):
    # same test as above but slight variation in the max_queries_dict
    # specify rate limits for "test" and "azure-open" apis and some models,
    # but only specify overall rate limit for "gemini"
    # also add other key which should be ignored but
    # present in the group_experiment_prompts
    max_queries_dict = {
        "test": {"model1": 160, "model2": 26},
        "gemini": 10,
        "azure-openai": {"default": 25, "gpt3": 300, "gpt4": 250},
        "ignored": 1,
        "ignored-with-models": {"default": 5, "ignored-model": 19},
    }

    # create a settings object
    settings = Settings(
        data_folder="experiment_pipeline",
        max_queries=50,
        max_attempts=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("larger_no_groups.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "ignored": {
            "prompt_dicts": [],
            "rate_limit": 1,
        },
        "ignored-with-models": {
            "prompt_dicts": [],
            "rate_limit": 5,
        },
        "ignored-with-models-ignored-model": {
            "prompt_dicts": [],
            "rate_limit": 19,
        },
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 6", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 7", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 8", "api": "test"},
            ],
            "rate_limit": 50,
        },
        "test-model1": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 3", "api": "test", "model_name": "model1"},
            ],
            "rate_limit": 160,
        },
        "test-model2": {
            "prompt_dicts": [
                {"prompt": "test prompt 5", "api": "test", "model_name": "model2"}
            ],
            "rate_limit": 26,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 3", "api": "gemini"},
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 10,
        },
        "azure-openai-gpt3": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
            ],
            "rate_limit": 300,
        },
        "azure-openai-gpt4": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
            ],
            "rate_limit": 250,
        },
        "azure-openai": {
            "prompt_dicts": [
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 25,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "ignored": "0 queries at 1 queries per minute",
        "test": "5 queries at 50 queries per minute",
        "test-model1": "2 queries at 160 queries per minute",
        "test-model2": "1 queries at 26 queries per minute",
        "gemini": "5 queries at 10 queries per minute",
        "azure-openai": "3 queries at 25 queries per minute",
        "azure-openai-gpt3": "3 queries at 300 queries per minute",
        "azure-openai-gpt4": "2 queries at 250 queries per minute",
        "ignored-with-models": "0 queries at 5 queries per minute",
        "ignored-with-models-ignored-model": "0 queries at 19 queries per minute",
    }


def test_experiment_grouped_prompts_default_with_groups(temporary_data_folders):
    # create a settings object without max_queries_dict (i.e. it's {} by default)
    settings = Settings(
        data_folder="experiment_pipeline", max_queries=50, max_attempts=5
    )
    assert settings.max_queries_dict == {}

    # parallel is by default False too but we will group the prompts when calling the attribute
    assert settings.parallel is False

    # create an experiment object
    experiment = Experiment("larger_with_groups.jsonl", settings=settings)

    # since we haven't passed in a max_queries_dict, the grouped prompts will be first grouped
    # by the "group" and then by the "api" key
    assert experiment.grouped_experiment_prompts == {
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
            ],
            "rate_limit": 50,
        },
        "group1": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 3",
                    "api": "test",
                    "model_name": "model1",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 5",
                    "api": "test",
                    "model_name": "model2",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 6",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group1",
                },
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group1",
                },
                {"prompt": "gemini prompt 3", "api": "gemini", "group": "group1"},
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                    "group": "group1",
                },
            ],
            "rate_limit": 50,
        },
        "group2": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 7",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group2",
                },
                {"prompt": "test prompt 8", "api": "test", "group": "group2"},
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group2",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                    "group": "group2",
                },
            ],
            "rate_limit": 50,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 50,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 50,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "test": "3 queries at 50 queries per minute",
        "group1": "8 queries at 50 queries per minute",
        "group2": "4 queries at 50 queries per minute",
        "gemini": "2 queries at 50 queries per minute",
        "azure-openai": "4 queries at 50 queries per minute",
    }


def test_experiment_grouped_prompts_apis_only_with_groups(temporary_data_folders):
    # specify rate limits for "test" and "gemini", but leave "azure-openai" and groups to default
    # also add another key which should be ignored but present in the group_experiment_prompts
    max_queries_dict = {"test": 100, "gemini": 200, "ignored": 10}

    # create a settings object
    settings = Settings(
        data_folder="experiment_pipeline",
        max_queries=50,
        max_attempts=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("larger_with_groups.jsonl", settings=settings)

    # remember "group" keys take precedence over "api" keys
    assert experiment.grouped_experiment_prompts == {
        "ignored": {
            "prompt_dicts": [],
            "rate_limit": 10,
        },
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
            ],
            "rate_limit": 100,
        },
        "group1": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 3",
                    "api": "test",
                    "model_name": "model1",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 5",
                    "api": "test",
                    "model_name": "model2",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 6",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group1",
                },
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group1",
                },
                {"prompt": "gemini prompt 3", "api": "gemini", "group": "group1"},
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                    "group": "group1",
                },
            ],
            "rate_limit": 50,
        },
        "group2": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 7",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group2",
                },
                {"prompt": "test prompt 8", "api": "test", "group": "group2"},
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group2",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                    "group": "group2",
                },
            ],
            "rate_limit": 50,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 200,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 50,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "test": "3 queries at 100 queries per minute",
        "group1": "8 queries at 50 queries per minute",
        "group2": "4 queries at 50 queries per minute",
        "gemini": "2 queries at 200 queries per minute",
        "azure-openai": "4 queries at 50 queries per minute",
        "ignored": "0 queries at 10 queries per minute",
    }


def test_experiment_grouped_prompts_groups_only_with_groups(temporary_data_folders):
    # specify rate limits for "group1" but everything else to default
    # also add another key which should be ignored but present in the group_experiment_prompts
    max_queries_dict = {"group1": 100, "ignored": 10}

    # create a settings object
    settings = Settings(
        data_folder="experiment_pipeline",
        max_queries=50,
        max_attempts=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("larger_with_groups.jsonl", settings=settings)

    # remember "group" keys take precedence over "api" keys
    assert experiment.grouped_experiment_prompts == {
        "ignored": {
            "prompt_dicts": [],
            "rate_limit": 10,
        },
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
            ],
            "rate_limit": 50,
        },
        "group1": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 3",
                    "api": "test",
                    "model_name": "model1",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 5",
                    "api": "test",
                    "model_name": "model2",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 6",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group1",
                },
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group1",
                },
                {"prompt": "gemini prompt 3", "api": "gemini", "group": "group1"},
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                    "group": "group1",
                },
            ],
            "rate_limit": 100,
        },
        "group2": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 7",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group2",
                },
                {"prompt": "test prompt 8", "api": "test", "group": "group2"},
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group2",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                    "group": "group2",
                },
            ],
            "rate_limit": 50,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 50,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 50,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "test": "3 queries at 50 queries per minute",
        "group1": "8 queries at 100 queries per minute",
        "group2": "4 queries at 50 queries per minute",
        "gemini": "2 queries at 50 queries per minute",
        "azure-openai": "4 queries at 50 queries per minute",
        "ignored": "0 queries at 10 queries per minute",
    }


def test_experiment_grouped_prompts_apis_and_models_with_groups(temporary_data_folders):
    # specify rate limits for "test" and "gemini" apis and some models,
    # but only specify overall rate limit for "azure-openai"
    # also add other key which should be ignored but
    # present in the group_experiment_prompts
    max_queries_dict = {
        "test": {"model2": 100, "model1": 200},
        "gemini": {"default": 25, "gemini-pro": 20},
        "azure-openai": 10,
        "ignored": 10,
        "ignored-with-models": {"default": 15, "ignored-model": 5},
    }

    # create a settings object
    settings = Settings(
        data_folder="experiment_pipeline",
        max_queries=50,
        max_attempts=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("larger_with_groups.jsonl", settings=settings)

    # remember "group" keys take precedence over "api" keys
    assert experiment.grouped_experiment_prompts == {
        "ignored": {
            "prompt_dicts": [],
            "rate_limit": 10,
        },
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
            ],
            "rate_limit": 50,
        },
        "test-model1": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
            ],
            "rate_limit": 200,
        },
        "test-model2": {
            "prompt_dicts": [],
            "rate_limit": 100,
        },
        "group1": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 3",
                    "api": "test",
                    "model_name": "model1",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 5",
                    "api": "test",
                    "model_name": "model2",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 6",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group1",
                },
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group1",
                },
                {"prompt": "gemini prompt 3", "api": "gemini", "group": "group1"},
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                    "group": "group1",
                },
            ],
            "rate_limit": 50,
        },
        "group2": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 7",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group2",
                },
                {"prompt": "test prompt 8", "api": "test", "group": "group2"},
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group2",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                    "group": "group2",
                },
            ],
            "rate_limit": 50,
        },
        "gemini": {
            "prompt_dicts": [
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 25,
        },
        "gemini-gemini-pro": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
            ],
            "rate_limit": 20,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 10,
        },
        "ignored-with-models": {
            "prompt_dicts": [],
            "rate_limit": 15,
        },
        "ignored-with-models-ignored-model": {
            "prompt_dicts": [],
            "rate_limit": 5,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "test": "2 queries at 50 queries per minute",
        "group1": "8 queries at 50 queries per minute",
        "group2": "4 queries at 50 queries per minute",
        "test-model1": "1 queries at 200 queries per minute",
        "test-model2": "0 queries at 100 queries per minute",
        "gemini": "1 queries at 25 queries per minute",
        "gemini-gemini-pro": "1 queries at 20 queries per minute",
        "azure-openai": "4 queries at 10 queries per minute",
        "ignored": "0 queries at 10 queries per minute",
        "ignored-with-models": "0 queries at 15 queries per minute",
        "ignored-with-models-ignored-model": "0 queries at 5 queries per minute",
    }


def test_experiment_grouped_prompts_groups_and_models_with_groups(
    temporary_data_folders,
):
    # specify rate limits for "group1" and some models,
    # and overall rate limit for "group2"
    # also add other key which should be ignored but
    # present in the group_experiment_prompts
    max_queries_dict = {
        "group1": {"default": 10, "gpt3": 100, "model2": 200},
        "group2": 75,
        "ignored": 10,
        "ignored-with-models": {"default": 15, "ignored-model": 5},
    }

    # create a settings object
    settings = Settings(
        data_folder="experiment_pipeline",
        max_queries=50,
        max_attempts=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("larger_with_groups.jsonl", settings=settings)

    # remember "group" keys take precedence over "api" keys
    assert experiment.grouped_experiment_prompts == {
        "ignored": {
            "prompt_dicts": [],
            "rate_limit": 10,
        },
        "ignored-with-models": {
            "prompt_dicts": [],
            "rate_limit": 15,
        },
        "ignored-with-models-ignored-model": {
            "prompt_dicts": [],
            "rate_limit": 5,
        },
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
            ],
            "rate_limit": 50,
        },
        "group1": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 3",
                    "api": "test",
                    "model_name": "model1",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 6",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group1",
                },
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group1",
                },
                {"prompt": "gemini prompt 3", "api": "gemini", "group": "group1"},
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                    "group": "group1",
                },
            ],
            "rate_limit": 10,
        },
        "group1-model2": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 5",
                    "api": "test",
                    "model_name": "model2",
                    "group": "group1",
                },
            ],
            "rate_limit": 200,
        },
        "group1-gpt3": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
            ],
            "rate_limit": 100,
        },
        "group2": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 7",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group2",
                },
                {"prompt": "test prompt 8", "api": "test", "group": "group2"},
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group2",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                    "group": "group2",
                },
            ],
            "rate_limit": 75,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 50,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 50,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "test": "3 queries at 50 queries per minute",
        "group1": "5 queries at 10 queries per minute",
        "group2": "4 queries at 75 queries per minute",
        "gemini": "2 queries at 50 queries per minute",
        "azure-openai": "4 queries at 50 queries per minute",
        "ignored": "0 queries at 10 queries per minute",
        "group1-model2": "1 queries at 200 queries per minute",
        "group1-gpt3": "2 queries at 100 queries per minute",
        "ignored-with-models": "0 queries at 15 queries per minute",
        "ignored-with-models-ignored-model": "0 queries at 5 queries per minute",
    }


def test_experiment_grouped_prompts_all_with_groups(temporary_data_folders):
    # specify rate limits for "test" and "gemini" apis and some models,
    # as well as limits for "group1" and some models,
    # and overall rate limit for "group2" and "azure-openai"
    # also add other key which should be ignored but
    # present in the group_experiment_prompts
    max_queries_dict = {
        "test": {"model2": 100, "model1": 200},
        "gemini": {"default": 25, "gemini-pro": 20},
        "group1": {"default": 10, "gpt3": 100, "model2": 200},
        "group2": 75,
        "azure-openai": 10,
        "ignored": 10,
        "ignored-with-models": {"default": 15, "ignored-model": 5},
    }

    # create a settings object
    settings = Settings(
        data_folder="experiment_pipeline",
        max_queries=50,
        max_attempts=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("larger_with_groups.jsonl", settings=settings)

    # remember "group" keys take precedence over "api" keys
    assert experiment.grouped_experiment_prompts == {
        "ignored": {
            "prompt_dicts": [],
            "rate_limit": 10,
        },
        "test": {
            "prompt_dicts": [
                {"prompt": "test prompt 2", "api": "test"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
            ],
            "rate_limit": 50,
        },
        "test-model1": {
            "prompt_dicts": [
                {"prompt": "test prompt 1", "api": "test", "model_name": "model1"},
            ],
            "rate_limit": 200,
        },
        "test-model2": {
            "prompt_dicts": [],
            "rate_limit": 100,
        },
        "group1": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 3",
                    "api": "test",
                    "model_name": "model1",
                    "group": "group1",
                },
                {
                    "prompt": "test prompt 6",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group1",
                },
                {
                    "prompt": "gemini prompt 1",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group1",
                },
                {"prompt": "gemini prompt 3", "api": "gemini", "group": "group1"},
                {
                    "prompt": "azure-openai prompt 6",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                    "group": "group1",
                },
            ],
            "rate_limit": 10,
        },
        "group1-model2": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 5",
                    "api": "test",
                    "model_name": "model2",
                    "group": "group1",
                },
            ],
            "rate_limit": 200,
        },
        "group1-gpt3": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 1",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
                {
                    "prompt": "azure-openai prompt 3",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                    "group": "group1",
                },
            ],
            "rate_limit": 100,
        },
        "group2": {
            "prompt_dicts": [
                {
                    "prompt": "test prompt 7",
                    "api": "test",
                    "model_name": "model3",
                    "group": "group2",
                },
                {"prompt": "test prompt 8", "api": "test", "group": "group2"},
                {
                    "prompt": "gemini prompt 2",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                    "group": "group2",
                },
                {
                    "prompt": "azure-openai prompt 8",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                    "group": "group2",
                },
            ],
            "rate_limit": 75,
        },
        "gemini": {
            "prompt_dicts": [
                {"prompt": "gemini prompt 5", "api": "gemini"},
            ],
            "rate_limit": 25,
        },
        "gemini-gemini-pro": {
            "prompt_dicts": [
                {
                    "prompt": "gemini prompt 4",
                    "api": "gemini",
                    "model_name": "gemini-pro",
                },
            ],
            "rate_limit": 20,
        },
        "azure-openai": {
            "prompt_dicts": [
                {
                    "prompt": "azure-openai prompt 2",
                    "api": "azure-openai",
                    "model_name": "gpt4",
                },
                {"prompt": "azure-openai prompt 4", "api": "azure-openai"},
                {
                    "prompt": "azure-openai prompt 5",
                    "api": "azure-openai",
                    "model_name": "gpt3",
                },
                {
                    "prompt": "azure-openai prompt 7",
                    "api": "azure-openai",
                    "model_name": "gpt3.5",
                },
            ],
            "rate_limit": 10,
        },
        "ignored-with-models": {
            "prompt_dicts": [],
            "rate_limit": 15,
        },
        "ignored-with-models-ignored-model": {
            "prompt_dicts": [],
            "rate_limit": 5,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "test": "2 queries at 50 queries per minute",
        "group1": "5 queries at 10 queries per minute",
        "group2": "4 queries at 75 queries per minute",
        "group1-model2": "1 queries at 200 queries per minute",
        "group1-gpt3": "2 queries at 100 queries per minute",
        "test-model1": "1 queries at 200 queries per minute",
        "test-model2": "0 queries at 100 queries per minute",
        "gemini": "1 queries at 25 queries per minute",
        "gemini-gemini-pro": "1 queries at 20 queries per minute",
        "azure-openai": "4 queries at 10 queries per minute",
        "ignored": "0 queries at 10 queries per minute",
        "ignored-with-models": "0 queries at 15 queries per minute",
        "ignored-with-models-ignored-model": "0 queries at 5 queries per minute",
    }
