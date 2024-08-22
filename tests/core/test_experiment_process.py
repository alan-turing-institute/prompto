import json
import logging
import os

import pytest

from prompto.experiment import Experiment
from prompto.settings import Settings

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_process(
    temporary_data_folder_for_processing: None,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2, max_queries=200)
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    assert experiment.completed_responses == []
    assert not os.path.isdir(experiment.output_folder)

    result, avg_query_proc_time = await experiment.process()

    # assert that the output folder was created and input file was moved to it
    assert os.path.isdir(experiment.output_folder)
    assert not os.path.isfile("data/input/test_experiment.jsonl")
    assert len(os.listdir("data/output/test_experiment")) == 3
    # assert created files in output
    assert os.path.isfile(
        f"data/output/test_experiment/{experiment.start_time}-completed-test_experiment.jsonl"
    )
    assert os.path.isfile(
        f"data/output/test_experiment/{experiment.start_time}-input-test_experiment.jsonl"
    )
    assert os.path.isfile(
        f"data/output/test_experiment/{experiment.start_time}-log-test_experiment.txt"
    )

    # check processing time
    assert isinstance(avg_query_proc_time, float)
    assert avg_query_proc_time > 0

    # check result
    assert len(result) == 6
    assert experiment.completed_responses == result

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert responses == result

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 6 queries at 200 QPM with RI of 0.3s (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Sending 1 queries at 200 QPM with RI of 0.3s (attempt 2/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 2/2)"
    assert print_msg in captured.err

    # check log messages
    log_msg = "Processing experiment: test_experiment.jsonl.."
    assert log_msg in caplog.text
    log_msg = (
        "Moving data/input/test_experiment.jsonl to "
        "data/output/test_experiment as "
        "data/output/test_experiment/"
        f"{experiment.start_time}-input-test_experiment.jsonl"
    )
    log_msg = "Sending 6 queries..."
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=1, id=0)\n"
        "Prompt: test prompt 1...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=2, id=NA) on attempt 1 of 2: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error with model test (i=3, id=1)\n"
        "Prompt: test prompt 3...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    log_msg = (
        "Response received for model test (i=4, id=2)\n"
        "Prompt: test prompt 4...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=5, id=3)\n"
        "Prompt: test prompt 5...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=6, id=4)\n"
        "Prompt: test prompt 6...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Retrying 1 failed queries - attempt 2 of 2..."
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=1, id=NA) after maximum 2 attempts: "
        "Exception - This is a test error which we should handle and return"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error with model test (i=1, id=NA)\n"
        "Prompt: test prompt 2...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    log_msg = "Maximum attempts reached. Exiting..."
    assert log_msg in caplog.text
    log_msg = "Completed experiment: test_experiment.jsonl! "
    assert log_msg in caplog.text
    log_msg = "Experiment processing time: "
    assert log_msg in caplog.text
    log_msg = "Average time per query: "
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_process_with_max_queries_dict(
    temporary_data_folder_for_processing: None, caplog, capsys
):
    caplog.set_level(logging.INFO)
    settings = Settings(
        data_folder="data",
        max_attempts=2,
        max_queries_dict={"test": {"model1": 200, "model2": 300}},
        parallel=True,
    )
    experiment = Experiment("test_experiment.jsonl", settings=settings)

    assert experiment.completed_responses == []
    assert not os.path.isdir(experiment.output_folder)

    result, avg_query_proc_time = await experiment.process()

    # assert that the output folder was created and input file was moved to it
    assert os.path.isdir(experiment.output_folder)
    assert not os.path.isfile("data/input/test_experiment.jsonl")
    assert len(os.listdir("data/output/test_experiment")) == 3
    # assert created files in output
    assert os.path.isfile(
        f"data/output/test_experiment/{experiment.start_time}-completed-test_experiment.jsonl"
    )
    assert os.path.isfile(
        f"data/output/test_experiment/{experiment.start_time}-input-test_experiment.jsonl"
    )
    assert os.path.isfile(
        f"data/output/test_experiment/{experiment.start_time}-log-test_experiment.txt"
    )

    # check processing time
    assert isinstance(avg_query_proc_time, float)
    assert avg_query_proc_time > 0

    # check result
    assert len(result) == 6
    assert experiment.completed_responses == result

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert responses == result

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = (
        "Sending 0 queries at 10 QPM with RI of 6.0s for group 'test' (attempt 1/2)"
    )
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test' (attempt 1/2)"
    print_msg = "Sending 3 queries at 200 QPM with RI of 0.3s for group 'test-model1' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test-model1' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Sending 3 queries at 300 QPM with RI of 0.2s for group 'test-model2' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test-model2' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Sending 1 queries at 200 QPM with RI of 0.3s for group 'test-model1' (attempt 2/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test-model1' (attempt 2/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for all groups to complete"
    assert print_msg in captured.err

    # check log messages
    log_msg = "Processing experiment: test_experiment.jsonl.."
    assert log_msg in caplog.text
    log_msg = (
        "Moving data/input/test_experiment.jsonl to "
        "data/output/test_experiment as "
        "data/output/test_experiment/"
        f"{experiment.start_time}-input-test_experiment.jsonl"
    )
    log_msg = "Sending 6 queries in parallel by grouping prompts..."
    assert log_msg in caplog.text
    log_msg = f"Queries per group: {experiment.grouped_experiment_prompts_summary()}"
    assert log_msg in caplog.text
    log_msg = "No remaining failed queries for group 'test'"
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=1, id=0)\n"
        "Prompt: test prompt 1...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=2, id=NA) on attempt 1 of 2: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error with model test (i=3, id=1)\n"
        "Prompt: test prompt 3...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    log_msg = (
        "Response received for model test (i=1, id=2)\n"
        "Prompt: test prompt 4...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=3)\n"
        "Prompt: test prompt 5...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=3, id=4)\n"
        "Prompt: test prompt 6...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "No remaining failed queries for group 'test-model2'"
    assert log_msg in caplog.text
    log_msg = "Retrying 1 failed queries for group 'test-model1' - attempt 2 of 2..."
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=1, id=NA) after maximum 2 attempts: "
        "Exception - This is a test error which we should handle and return"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error with model test (i=1, id=NA)\n"
        "Prompt: test prompt 2...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    log_msg = "Maximum attempts reached for group 'test-model1'. Exiting..."
    assert log_msg in caplog.text
    log_msg = "Completed experiment: test_experiment.jsonl! "
    assert log_msg in caplog.text
    log_msg = "Experiment processing time: "
    assert log_msg in caplog.text
    log_msg = "Average time per query: "
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_process_with_groups(
    temporary_data_folder_for_processing: None,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
):
    caplog.set_level(logging.INFO)
    settings = Settings(data_folder="data", max_attempts=2, max_queries=300)
    experiment = Experiment("test_experiment_with_groups.jsonl", settings=settings)

    assert experiment.completed_responses == []
    assert not os.path.isdir(experiment.output_folder)

    result, avg_query_proc_time = await experiment.process()

    # assert that the output folder was created and input file was moved to it
    assert os.path.isdir(experiment.output_folder)
    assert not os.path.isfile("data/input/test_experiment_with_groups.jsonl")
    assert len(os.listdir("data/output/test_experiment_with_groups")) == 3
    # assert created files in output
    assert os.path.isfile(
        f"data/output/test_experiment_with_groups/{experiment.start_time}-completed-test_experiment_with_groups.jsonl"
    )
    assert os.path.isfile(
        f"data/output/test_experiment_with_groups/{experiment.start_time}-input-test_experiment_with_groups.jsonl"
    )
    assert os.path.isfile(
        f"data/output/test_experiment_with_groups/{experiment.start_time}-log-test_experiment_with_groups.txt"
    )

    # check processing time
    assert isinstance(avg_query_proc_time, float)
    assert avg_query_proc_time > 0

    # check result
    assert len(result) == 6
    assert experiment.completed_responses == result

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert responses == result

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = "Sending 6 queries at 300 QPM with RI of 0.2s (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Sending 1 queries at 300 QPM with RI of 0.2s (attempt 2/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses (attempt 2/2)"
    assert print_msg in captured.err

    # check log messages
    log_msg = "Processing experiment: test_experiment_with_groups.jsonl.."
    assert log_msg in caplog.text
    log_msg = (
        "Moving data/input/test_experiment_with_groups.jsonl to "
        "data/output/test_experiment_with_groups as "
        "data/output/test_experiment_with_groups/"
        f"{experiment.start_time}-input-test_experiment_with_groups.jsonl"
    )
    log_msg = "Sending 6 queries..."
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=1, id=0)\n"
        "Prompt: test prompt 1...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=2, id=NA) on attempt 1 of 2: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=3, id=1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error with model test (i=3, id=1)\n"
        "Prompt: test prompt 3...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    log_msg = (
        "Response received for model test (i=4, id=2)\n"
        "Prompt: test prompt 4...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=5, id=3)\n"
        "Prompt: test prompt 5...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=6, id=4)\n"
        "Prompt: test prompt 6...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "Retrying 1 failed queries - attempt 2 of 2..."
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=1, id=NA) after maximum 2 attempts: "
        "Exception - This is a test error which we should handle and return"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error with model test (i=1, id=NA)\n"
        "Prompt: test prompt 2...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    log_msg = "Maximum attempts reached. Exiting..."
    assert log_msg in caplog.text
    log_msg = "Completed experiment: test_experiment_with_groups.jsonl! "
    assert log_msg in caplog.text
    log_msg = "Experiment processing time: "
    assert log_msg in caplog.text
    log_msg = "Average time per query: "
    assert log_msg in caplog.text


@pytest.mark.asyncio
async def test_process_with_max_queries_dict_and_groups(
    temporary_data_folder_for_processing, caplog, capsys
):
    caplog.set_level(logging.INFO)
    settings = Settings(
        data_folder="data",
        max_attempts=2,
        max_queries_dict={"test": {"model1": 200, "model2": 300}, "group1": 500},
        parallel=True,
    )
    experiment = Experiment("test_experiment_with_groups.jsonl", settings=settings)

    assert experiment.completed_responses == []
    assert not os.path.isdir(experiment.output_folder)

    result, avg_query_proc_time = await experiment.process()

    # assert that the output folder was created and input file was moved to it
    assert os.path.isdir(experiment.output_folder)
    assert not os.path.isfile("data/input/test_experiment_with_groups.jsonl")
    assert len(os.listdir("data/output/test_experiment_with_groups")) == 3
    # assert created files in output
    assert os.path.isfile(
        f"data/output/test_experiment_with_groups/{experiment.start_time}-completed-test_experiment_with_groups.jsonl"
    )
    assert os.path.isfile(
        f"data/output/test_experiment_with_groups/{experiment.start_time}-input-test_experiment_with_groups.jsonl"
    )
    assert os.path.isfile(
        f"data/output/test_experiment_with_groups/{experiment.start_time}-log-test_experiment_with_groups.txt"
    )

    # check processing time
    assert isinstance(avg_query_proc_time, float)
    assert avg_query_proc_time > 0

    # check result
    assert len(result) == 6
    assert experiment.completed_responses == result

    # check that the response is saved to the output file
    assert os.path.exists(experiment.output_completed_file_path)
    with open(experiment.output_completed_file_path, "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert responses == result

    # check the content printed to the console (tqdm progress bar)
    captured = capsys.readouterr()
    print_msg = (
        "Sending 0 queries at 10 QPM with RI of 6.0s for group 'test' (attempt 1/2)"
    )
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test' (attempt 1/2)"
    print_msg = "Sending 2 queries at 200 QPM with RI of 0.3s for group 'test-model1' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test-model1' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Sending 2 queries at 300 QPM with RI of 0.2s for group 'test-model2' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test-model2' (attempt 1/2)"
    print_msg = (
        "Sending 2 queries at 500 QPM with RI of 0.12s for group 'group1' (attempt 1/2)"
    )
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'group1' (attempt 1/2)"
    assert print_msg in captured.err
    print_msg = "Sending 1 queries at 200 QPM with RI of 0.3s for group 'test-model1' (attempt 2/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for responses for group 'test-model1' (attempt 2/2)"
    assert print_msg in captured.err
    print_msg = "Waiting for all groups to complete"
    assert print_msg in captured.err

    # check log messages
    log_msg = "Processing experiment: test_experiment_with_groups.jsonl.."
    assert log_msg in caplog.text
    log_msg = (
        "Moving data/input/test_experiment_with_groups.jsonl to "
        "data/output/test_experiment_with_groups as "
        "data/output/test_experiment_with_groups/"
        f"{experiment.start_time}-input-test_experiment_with_groups.jsonl"
    )
    log_msg = "Sending 6 queries in parallel by grouping prompts..."
    assert log_msg in caplog.text
    log_msg = f"Queries per group: {experiment.grouped_experiment_prompts_summary()}"
    assert log_msg in caplog.text
    log_msg = "No remaining failed queries for group 'test'"
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=1, id=0)\n"
        "Prompt: test prompt 1...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=2, id=NA) on attempt 1 of 2: "
        "Exception - This is a test error which we should handle and return. "
        "Adding to the queue to try again later..."
    )
    assert log_msg in caplog.text
    log_msg = "Error (i=1, id=1): ValueError - This is a test error which we should handle and return"
    assert log_msg in caplog.text
    log_msg = (
        "Error with model test (i=1, id=1)\n"
        "Prompt: test prompt 3...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    log_msg = (
        "Response received for model test (i=1, id=2)\n"
        "Prompt: test prompt 4...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=3)\n"
        "Prompt: test prompt 5...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Response received for model test (i=2, id=4)\n"
        "Prompt: test prompt 6...\n"
        "Response: This is a test response...\n"
    )
    assert log_msg in caplog.text
    log_msg = "No remaining failed queries for group 'group1'"
    assert log_msg in caplog.text
    log_msg = "No remaining failed queries for group 'test-model2'"
    assert log_msg in caplog.text
    log_msg = "Retrying 1 failed queries for group 'test-model1' - attempt 2 of 2..."
    assert log_msg in caplog.text
    log_msg = (
        "Error (i=1, id=NA) after maximum 2 attempts: "
        "Exception - This is a test error which we should handle and return"
    )
    assert log_msg in caplog.text
    log_msg = (
        "Error with model test (i=1, id=NA)\n"
        "Prompt: test prompt 2...\n"
        "Error: This is a test error which we should handle and return\n"
    )
    log_msg = "Maximum attempts reached for group 'test-model1'. Exiting..."
    assert log_msg in caplog.text
    log_msg = "Completed experiment: test_experiment_with_groups.jsonl! "
    assert log_msg in caplog.text
    log_msg = "Experiment processing time: "
    assert log_msg in caplog.text
    log_msg = "Average time per query: "
    assert log_msg in caplog.text
