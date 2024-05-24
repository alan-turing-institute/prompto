import logging
import os

import pytest

from prompto.experiment_processing import Experiment, ExperimentPipeline
from prompto.settings import Settings


def test_experiment_pipeline_init_errors(temporary_data_folders):
    # raise TypeError if not passing in settings
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        ExperimentPipeline()


def test_experiment_pipeline_init(temporary_data_folders):
    # initialise ExperimentPipeline object
    settings = Settings()
    pipeline = ExperimentPipeline(settings=settings)

    # check the attributes of the pipeline object
    assert pipeline.settings == settings
    assert pipeline.average_per_query_processing_times == []
    assert pipeline.overall_avg_proc_times == 0.0
    assert pipeline.experiment_files == []


def test_update_experiment_files(temporary_data_folders):
    # initialise ExperimentPipeline object
    settings = Settings(data_folder="experiment_pipeline")
    pipeline = ExperimentPipeline(settings=settings)

    # check that the experiment_files attribute is empty initially
    assert pipeline.experiment_files == []

    # call the update_experiment_files method
    pipeline.update_experiment_files()
    assert pipeline.experiment_files == ["first.jsonl", "second.jsonl", "larger.jsonl"]

    # remove second.jsonl from the folder
    os.remove("experiment_pipeline/input/second.jsonl")
    pipeline.update_experiment_files()
    assert pipeline.experiment_files == ["first.jsonl", "larger.jsonl"]


def test_log_estimate(temporary_data_folders, caplog):
    caplog.set_level(logging.INFO)

    # initialise ExperimentPipeline object
    settings = Settings(data_folder="experiment_pipeline")
    pipeline = ExperimentPipeline(settings=settings)

    # make first.jsonl an Experiment object
    first = Experiment("first.jsonl", settings=settings)
    assert first.number_queries == 3

    # raise error if not passing in experiment
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        pipeline.log_estimate()

    output_experiment_dir = "experiment_pipeline/output/first"
    # create output folder for experiment so a log can be written
    os.makedirs(output_experiment_dir)
    # if overall_avg_proc_times is 0.0, log_estimate should be unknown
    pipeline.log_estimate(experiment=first)
    expected_log = "Next experiment: first.jsonl, Number of queries: 3, Estimated completion time: [unknown], Estimated completion by: [unknown]"
    assert expected_log in caplog.text
    # check the log file was created (should only be one file in the output folder)
    assert len(os.listdir(output_experiment_dir)) == 1
    # check the expected log message is in the log file
    log_file_name = os.path.join(
        output_experiment_dir, os.listdir(output_experiment_dir)[0]
    )
    assert expected_log in open(log_file_name).read()
    # delete the output experiment folder (and its contents)
    os.remove(log_file_name)
    os.rmdir(output_experiment_dir)

    # update the overall average query processing time
    pipeline.overall_avg_proc_times = 0.5
    # create output folder for experiment so a log can be written
    os.makedirs(output_experiment_dir)
    # if overall_avg_proc_times is not 0.0, log_estimate should be calculated by multiplying the number of queries
    pipeline.log_estimate(experiment=first)
    expected_log = "Next experiment: first.jsonl, Number of queries: 3, Estimated completion time: 1.5, Estimated completion by:"
    assert expected_log in caplog.text
    # check the log file was created (should only be one file in the output folder)
    assert len(os.listdir(output_experiment_dir)) == 1
    # check the expected log message is in the log file
    assert (
        expected_log
        in open(
            os.path.join(output_experiment_dir, os.listdir(output_experiment_dir)[0])
        ).read()
    )


def test_log_progress(temporary_data_folders, caplog):
    caplog.set_level(logging.INFO)

    # initialise ExperimentPipeline object
    settings = Settings(data_folder="experiment_pipeline")
    pipeline = ExperimentPipeline(settings=settings)

    # make first.jsonl and second.jsonl  Experiment objects
    first = Experiment("first.jsonl", settings=settings)
    second = Experiment("second.jsonl", settings=settings)

    # raise error if not passing in experiment
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        pipeline.log_progress()

    # experiment_files should be empty initially
    assert pipeline.experiment_files == []

    # move first.jsonl to the output folder (simulating it being processed)
    output_experiment_dir = "experiment_pipeline/output/first"
    # create output folder for experiment so a log can be written
    os.makedirs(output_experiment_dir)
    os.rename(
        "experiment_pipeline/input/first.jsonl",
        os.path.join(output_experiment_dir, "first.jsonl"),
    )
    # call the log_progress method on first experiment
    pipeline.log_progress(experiment=first)
    # check the experiment_files attribute is updated (should only have second.jsonl)
    assert pipeline.experiment_files == ["second.jsonl", "larger.jsonl"]
    # check the log messages have been written
    assert "Completed experiment: first.jsonl!" in caplog.text
    assert "- Overall average time per query: 0.0" in caplog.text
    assert "- Remaining number of experiments: 2" in caplog.text
    assert "- Remaining experiments: ['second.jsonl', 'larger.jsonl']" in caplog.text

    # move second.jsonl to the output folder (simulating it being processed)
    output_experiment_dir = "experiment_pipeline/output/second"
    # create output folder for experiment so a log can be written
    os.makedirs(output_experiment_dir)
    os.rename(
        "experiment_pipeline/input/second.jsonl",
        os.path.join(output_experiment_dir, "second.jsonl"),
    )
    # manually update the overall average processing time
    pipeline.overall_avg_proc_times = 0.5
    # call the log_progress method on second experiment
    pipeline.log_progress(experiment=second)
    # check the experiment_files attribute is empty
    assert pipeline.experiment_files == ["larger.jsonl"]
    # check the log messages have been written
    assert "Completed experiment: second.jsonl!" in caplog.text
    assert "- Overall average time per query: 0.5" in caplog.text
    assert "- Remaining number of experiments: 1" in caplog.text
    assert "- Remaining experiments: ['larger.jsonl']" in caplog.text
