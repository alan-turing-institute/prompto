import logging

import pytest

from prompto.experiment import Experiment
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
    with pytest.raises(ValueError, match="Experiment file must be a jsonl or csv file"):
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
        f.write(
            '{"id": 0, "prompt": "test prompt 0", "api": "test", "model_name": "test_model"}\n'
        )
        f.write(
            '{"id": 1, "prompt": "test prompt 1", "api": "test", "model_name": "test_model"}\n'
        )

    # create an experiment object
    experiment = Experiment("test_in_input.jsonl", settings=settings)

    # check the experiment object has the correct attributes
    assert experiment.file_name == "test_in_input.jsonl"
    assert experiment.experiment_name == "test_in_input"
    assert experiment.settings == settings
    assert experiment.output_folder == "data/output/test_in_input"
    assert experiment.input_file_path == "data/input/test_in_input.jsonl"
    assert isinstance(experiment.creation_time, str)
    assert isinstance(experiment.start_time, str)
    assert (
        experiment.output_completed_jsonl_file_path
        == f"data/output/test_in_input/{experiment.start_time}-completed-test_in_input.jsonl"
    )
    assert (
        experiment.output_input_jsonl_file_out_path
        == f"data/output/test_in_input/{experiment.start_time}-input-test_in_input.jsonl"
    )
    assert experiment._experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test", "model_name": "test_model"},
        {"id": 1, "prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
    ]
    # check property getter for experiment_prompts
    assert experiment.experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test", "model_name": "test_model"},
        {"id": 1, "prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
    ]
    assert experiment.number_queries == 2
    assert (
        experiment.log_file
        == f"data/output/test_in_input/{experiment.start_time}-log-test_in_input.txt"
    )

    # test str method
    assert str(experiment) == "test_in_input.jsonl"

    # test that grouped experiments have not been created yet
    assert experiment._grouped_experiment_prompts == {}

    assert experiment.completed_responses == []


def test_experiment_grouped_prompts_simple(temporary_data_folders, caplog):
    caplog.set_level(logging.WARNING)

    # create a settings object
    settings = Settings(data_folder="data", max_queries=50, max_attempts=5)

    # create a jsonl file in the input folder (which is created when initialising Settings object)
    with open("data/input/test_in_input.jsonl", "w") as f:
        f.write(
            '{"id": 0, "prompt": "test prompt 0", "api": "test", "model_name": "test_model"}\n'
        )
        f.write(
            '{"id": 1, "prompt": "test prompt 1", "api": "test", "model_name": "test_model"}\n'
        )

    # create an experiment object
    experiment = Experiment("test_in_input.jsonl", settings=settings)

    # check the experiment_prompts attribute is correct
    assert experiment._experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test", "model_name": "test_model"},
        {"id": 1, "prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
    ]
    # check property getter for experiment_prompts
    assert experiment.experiment_prompts == [
        {"id": 0, "prompt": "test prompt 0", "api": "test", "model_name": "test_model"},
        {"id": 1, "prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
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

    # check the warning message was given for calling the grouped_experiment_prompts
    # attribute when the parallel attribute is False
    log_msg = (
        "The 'parallel' attribute in the Settings object is set to False, "
        "so grouping will not be used when processing the experiment prompts. "
        "Set 'parallel' to True to use grouping and parallel processing of prompts."
    )
    assert log_msg in caplog.text


def test_experiment_experiment_prompts_getter(temporary_data_folders):
    # create a settings object
    settings = Settings(data_folder="experiment_pipeline")

    # create an experiment object
    experiment = Experiment("first.jsonl", settings=settings)

    assert experiment._experiment_prompts == [
        {"prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 3", "api": "test", "model_name": "test_model"},
    ]
    assert experiment.experiment_prompts == [
        {"prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 3", "api": "test", "model_name": "test_model"},
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
        {"prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 3", "api": "test", "model_name": "test_model"},
    ]


def test_experiment_grouped_prompts_getter(temporary_data_folders):
    # create a settings object
    settings = Settings(data_folder="experiment_pipeline")

    # create an experiment object
    experiment = Experiment("first.jsonl", settings=settings)

    # check only created when it's called (initially it's an empty dict)
    assert experiment._grouped_experiment_prompts == {}
    assert experiment.experiment_prompts == [
        {"prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 3", "api": "test", "model_name": "test_model"},
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
        {"prompt": "test prompt 1", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
        {"prompt": "test prompt 3", "api": "test", "model_name": "test_model"},
    ]
    assert experiment.grouped_experiment_prompts == {
        "test": {"prompt_dicts": experiment.experiment_prompts, "rate_limit": 10}
    }

    # try to set the grouped_experiment_prompts attribute when it's been set
    with pytest.raises(
        AttributeError, match="Cannot set the grouped_experiment_prompts attribute"
    ):
        experiment.grouped_experiment_prompts = {"hello": "world"}


def test_experiment_grouped_prompts_default_no_groups(
    temporary_data_folder_for_grouping_prompts, caplog
):
    caplog.set_level(logging.WARNING)

    # create a settings object without max_queries_dict (i.e. it's {} by default)
    settings = Settings(data_folder="data", max_queries=50, max_attempts=5)
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
                {"prompt": "test prompt 3", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 5", "api": "test", "model_name": "model2"},
                {"prompt": "test prompt 6", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 7", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 8", "api": "test", "model_name": "test_model"},
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

    # check the warning message was given for calling the grouped_experiment_prompts
    # attribute when the parallel attribute is False
    log_msg = (
        "The 'parallel' attribute in the Settings object is set to False, "
        "so grouping will not be used when processing the experiment prompts. "
        "Set 'parallel' to True to use grouping and parallel processing of prompts."
    )
    assert log_msg in caplog.text


def test_experiment_grouped_prompts_apis_only_no_groups(
    temporary_data_folder_for_grouping_prompts,
):
    # specify rate limits for "test" and "gemini", but leave "azure-openai" to default
    # also add another key which should be ignored but present in the group_experiment_prompts
    max_queries_dict = {"test": 100, "gemini": 200, "ignored": 10}

    # create a settings object
    settings = Settings(
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
                {"prompt": "test prompt 3", "api": "test", "model_name": "model1"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 5", "api": "test", "model_name": "model2"},
                {"prompt": "test prompt 6", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 7", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 8", "api": "test", "model_name": "test_model"},
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
    temporary_data_folder_for_grouping_prompts,
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
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 6", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 7", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 8", "api": "test", "model_name": "test_model"},
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
    temporary_data_folder_for_grouping_prompts,
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
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
                {"prompt": "test prompt 4", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 6", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 7", "api": "test", "model_name": "model3"},
                {"prompt": "test prompt 8", "api": "test", "model_name": "test_model"},
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


def test_experiment_grouped_prompts_default_with_groups(
    temporary_data_folder_for_grouping_prompts, caplog
):
    caplog.set_level(logging.WARNING)

    # create a settings object without max_queries_dict (i.e. it's {} by default)
    settings = Settings(data_folder="data", max_queries=50, max_attempts=5)
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
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

    # check the warning message was given for calling the grouped_experiment_prompts
    # attribute when the parallel attribute is False
    log_msg = (
        "The 'parallel' attribute in the Settings object is set to False, "
        "so grouping will not be used when processing the experiment prompts. "
        "Set 'parallel' to True to use grouping and parallel processing of prompts."
    )
    assert log_msg in caplog.text


def test_experiment_grouped_prompts_apis_only_with_groups(
    temporary_data_folder_for_grouping_prompts,
):
    # specify rate limits for "test" and "gemini", but leave "azure-openai" and groups to default
    # also add another key which should be ignored but present in the group_experiment_prompts
    max_queries_dict = {"test": 100, "gemini": 200, "ignored": 10}

    # create a settings object
    settings = Settings(
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
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


def test_experiment_grouped_prompts_groups_only_with_groups(
    temporary_data_folder_for_grouping_prompts,
):
    # specify rate limits for "group1" but everything else to default
    # also add another key which should be ignored but present in the group_experiment_prompts
    max_queries_dict = {"group1": 100, "ignored": 10}

    # create a settings object
    settings = Settings(
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
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


def test_experiment_grouped_prompts_apis_and_models_with_groups(
    temporary_data_folder_for_grouping_prompts,
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
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
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
    temporary_data_folder_for_grouping_prompts,
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
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
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


def test_experiment_grouped_prompts_all_with_groups_v1(
    temporary_data_folder_for_grouping_prompts,
):
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
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
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


def test_experiment_grouped_prompts_all_with_groups_v2(
    temporary_data_folder_for_grouping_prompts,
):
    # same test as above but slight variation in the max_queries_dict
    # specify rate limits for "test" and "gemini" apis and some models,
    # as well as limits for "group1" and some models
    # also add other key which should be ignored but
    # present in the group_experiment_prompts
    max_queries_dict = {
        "test": {"model2": 100, "model1": 200},
        "gemini": {"default": 25, "gemini-pro": 20},
        "group1": {"default": 10, "gpt3": 100, "model2": 200},
        "ignored": 1,
        "ignored-with-models": {"default": 5, "ignored-model": 19},
    }

    # create a settings object
    settings = Settings(
        data_folder="data",
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
                {"prompt": "test prompt 2", "api": "test", "model_name": "test_model"},
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
            "rate_limit": 50,
        },
    }

    # test grouped_experiment_prompts_summary
    assert experiment.grouped_experiment_prompts_summary() == {
        "test": "2 queries at 50 queries per minute",
        "group1": "5 queries at 10 queries per minute",
        "group2": "4 queries at 50 queries per minute",
        "group1-model2": "1 queries at 200 queries per minute",
        "group1-gpt3": "2 queries at 100 queries per minute",
        "test-model1": "1 queries at 200 queries per minute",
        "test-model2": "0 queries at 100 queries per minute",
        "gemini": "1 queries at 25 queries per minute",
        "gemini-gemini-pro": "1 queries at 20 queries per minute",
        "azure-openai": "4 queries at 50 queries per minute",
        "ignored": "0 queries at 1 queries per minute",
        "ignored-with-models": "0 queries at 5 queries per minute",
        "ignored-with-models-ignored-model": "0 queries at 19 queries per minute",
    }


def test_rate_limit_docs_example_1(temporary_rate_limit_doc_examples):
    # create a settings object
    settings = Settings(
        max_queries=5,
        parallel=True,
    )
    assert settings.max_queries_dict == {}

    # create an experiment object
    experiment = Experiment("rate_limit_docs_example.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "gemini": {
            "prompt_dicts": [
                {
                    "id": 0,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 1,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 2,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 3,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 5,
        },
        "openai": {
            "prompt_dicts": [
                {
                    "id": 4,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 5,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 6,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 7,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 5,
        },
        "ollama": {
            "prompt_dicts": [
                {
                    "id": 8,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 9,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 10,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 11,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 5,
        },
    }

    assert experiment.grouped_experiment_prompts_summary() == {
        "gemini": "4 queries at 5 queries per minute",
        "openai": "4 queries at 5 queries per minute",
        "ollama": "4 queries at 5 queries per minute",
    }


def test_rate_limit_docs_example_2(temporary_rate_limit_doc_examples):
    max_queries_dict = {"openai": 20, "gemini": 10}
    # create a settings object
    settings = Settings(
        max_queries=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("rate_limit_docs_example.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "gemini": {
            "prompt_dicts": [
                {
                    "id": 0,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 1,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 2,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 3,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 10,
        },
        "openai": {
            "prompt_dicts": [
                {
                    "id": 4,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 5,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 6,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 7,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 20,
        },
        "ollama": {
            "prompt_dicts": [
                {
                    "id": 8,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 9,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 10,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 11,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 5,
        },
    }

    assert experiment.grouped_experiment_prompts_summary() == {
        "gemini": "4 queries at 10 queries per minute",
        "openai": "4 queries at 20 queries per minute",
        "ollama": "4 queries at 5 queries per minute",
    }


def test_rate_limit_docs_example_3(temporary_rate_limit_doc_examples):
    max_queries_dict = {
        "gemini": {"gemini-1.5-pro": 20},
        "openai": {"gpt4": 10, "gpt3.5-turbo": 20},
    }

    # create a settings object
    settings = Settings(
        max_queries=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("rate_limit_docs_example.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "gemini-gemini-1.5-pro": {
            "prompt_dicts": [
                {
                    "id": 2,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 3,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 20,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "id": 0,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 1,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 5,
        },
        "openai-gpt3.5-turbo": {
            "prompt_dicts": [
                {
                    "id": 4,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 5,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 20,
        },
        "openai": {
            "prompt_dicts": [],
            "rate_limit": 5,
        },
        "openai-gpt4": {
            "prompt_dicts": [
                {
                    "id": 6,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 7,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 10,
        },
        "ollama": {
            "prompt_dicts": [
                {
                    "id": 8,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 9,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 10,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 11,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 5,
        },
    }

    assert experiment.grouped_experiment_prompts_summary() == {
        "gemini": "2 queries at 5 queries per minute",
        "openai": "0 queries at 5 queries per minute",
        "gemini-gemini-1.5-pro": "2 queries at 20 queries per minute",
        "openai-gpt3.5-turbo": "2 queries at 20 queries per minute",
        "openai-gpt4": "2 queries at 10 queries per minute",
        "ollama": "4 queries at 5 queries per minute",
    }


def test_rate_limit_docs_example_4(temporary_rate_limit_doc_examples):
    max_queries_dict = {
        "gemini": {"default": 30, "gemini-1.5-pro": 20},
        "openai": {"gpt4": 10, "gpt3.5-turbo": 20},
        "ollama": 4,
    }

    # create a settings object
    settings = Settings(
        max_queries=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("rate_limit_docs_example.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "gemini-gemini-1.5-pro": {
            "prompt_dicts": [
                {
                    "id": 2,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 3,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 20,
        },
        "gemini": {
            "prompt_dicts": [
                {
                    "id": 0,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 1,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 30,
        },
        "openai-gpt3.5-turbo": {
            "prompt_dicts": [
                {
                    "id": 4,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 5,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 20,
        },
        "openai": {
            "prompt_dicts": [],
            "rate_limit": 5,
        },
        "openai-gpt4": {
            "prompt_dicts": [
                {
                    "id": 6,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 7,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 10,
        },
        "ollama": {
            "prompt_dicts": [
                {
                    "id": 8,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 9,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 10,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 11,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 4,
        },
    }

    assert experiment.grouped_experiment_prompts_summary() == {
        "gemini": "2 queries at 30 queries per minute",
        "openai": "0 queries at 5 queries per minute",
        "gemini-gemini-1.5-pro": "2 queries at 20 queries per minute",
        "openai-gpt3.5-turbo": "2 queries at 20 queries per minute",
        "openai-gpt4": "2 queries at 10 queries per minute",
        "ollama": "4 queries at 4 queries per minute",
    }


def test_rate_limit_docs_example_5(temporary_rate_limit_doc_examples):
    max_queries_dict = {"group1": 5, "group2": 10, "group3": 15}

    # create a settings object
    settings = Settings(
        max_queries=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("rate_limit_docs_example_groups.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "group1": {
            "prompt_dicts": [
                {
                    "id": 0,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of France?",
                    "group": "group1",
                },
                {
                    "id": 2,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of France?",
                    "group": "group1",
                },
                {
                    "id": 4,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of France?",
                    "group": "group1",
                },
                {
                    "id": 6,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of France?",
                    "group": "group1",
                },
            ],
            "rate_limit": 5,
        },
        "group2": {
            "prompt_dicts": [
                {
                    "id": 1,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of Germany?",
                    "group": "group2",
                },
                {
                    "id": 3,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of Germany?",
                    "group": "group2",
                },
                {
                    "id": 5,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of Germany?",
                    "group": "group2",
                },
                {
                    "id": 7,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of Germany?",
                    "group": "group2",
                },
            ],
            "rate_limit": 10,
        },
        "group3": {
            "prompt_dicts": [
                {
                    "id": 8,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of France?",
                    "group": "group3",
                },
                {
                    "id": 9,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of Germany?",
                    "group": "group3",
                },
                {
                    "id": 10,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of France?",
                    "group": "group3",
                },
                {
                    "id": 11,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of Germany?",
                    "group": "group3",
                },
            ],
            "rate_limit": 15,
        },
    }

    assert experiment.grouped_experiment_prompts_summary() == {
        "group1": "4 queries at 5 queries per minute",
        "group2": "4 queries at 10 queries per minute",
        "group3": "4 queries at 15 queries per minute",
    }


def test_rate_limit_docs_example_6(temporary_rate_limit_doc_examples):
    max_queries_dict = {"group1": 5, "group2": 10}

    # create a settings object
    settings = Settings(
        max_queries=5,
        parallel=True,
        max_queries_dict=max_queries_dict,
    )
    assert settings.max_queries_dict == max_queries_dict

    # create an experiment object
    experiment = Experiment("rate_limit_docs_example_groups_2.jsonl", settings=settings)

    assert experiment.grouped_experiment_prompts == {
        "gemini": {
            "prompt_dicts": [
                {
                    "id": 0,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 1,
                    "api": "gemini",
                    "model_name": "gemini-1.0-pro",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 2,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 3,
                    "api": "gemini",
                    "model_name": "gemini-1.5-pro",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 5,
        },
        "openai": {
            "prompt_dicts": [
                {
                    "id": 4,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 5,
                    "api": "openai",
                    "model_name": "gpt3.5-turbo",
                    "prompt": "What is the capital of Germany?",
                },
                {
                    "id": 6,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of France?",
                },
                {
                    "id": 7,
                    "api": "openai",
                    "model_name": "gpt4",
                    "prompt": "What is the capital of Germany?",
                },
            ],
            "rate_limit": 5,
        },
        "group1": {
            "prompt_dicts": [
                {
                    "id": 8,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of France?",
                    "group": "group1",
                },
                {
                    "id": 9,
                    "api": "ollama",
                    "model_name": "llama3",
                    "prompt": "What is the capital of Germany?",
                    "group": "group1",
                },
                {
                    "id": 10,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of France?",
                    "group": "group1",
                },
                {
                    "id": 11,
                    "api": "ollama",
                    "model_name": "mistral",
                    "prompt": "What is the capital of Germany?",
                    "group": "group1",
                },
            ],
            "rate_limit": 5,
        },
        "group2": {
            "prompt_dicts": [
                {
                    "id": 12,
                    "api": "ollama",
                    "model_name": "gemma",
                    "prompt": "What is the capital of France?",
                    "group": "group2",
                },
                {
                    "id": 13,
                    "api": "ollama",
                    "model_name": "gemma",
                    "prompt": "What is the capital of Germany?",
                    "group": "group2",
                },
                {
                    "id": 14,
                    "api": "ollama",
                    "model_name": "phi3",
                    "prompt": "What is the capital of France?",
                    "group": "group2",
                },
                {
                    "id": 15,
                    "api": "ollama",
                    "model_name": "phi3",
                    "prompt": "What is the capital of Germany?",
                    "group": "group2",
                },
            ],
            "rate_limit": 10,
        },
    }

    assert experiment.grouped_experiment_prompts_summary() == {
        "group1": "4 queries at 5 queries per minute",
        "group2": "4 queries at 10 queries per minute",
        "gemini": "4 queries at 5 queries per minute",
        "openai": "4 queries at 5 queries per minute",
    }
