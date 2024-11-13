import json
import os

from cli_test_helpers import shell


def test_run_experiment_entrypoint():
    result = shell("prompto_run_experiment --help")
    assert result.exit_code == 0


def test_run_experiment_no_inputs():
    result = shell("prompto_run_experiment")
    assert result.exit_code != 0
    assert "usage:" in result.stderr


def test_run_experiment_no_judge_in_input(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert (
        "Not creating judge file as one of judge_folder, judge or templates is None"
        in result.stderr
    )
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.jsonl..."
        in result.stderr
    )
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")


def test_run_experiment_no_judge_not_in_input_move(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file test-exp-not-in-input.jsonl "
        "--env-file some_file.env "
        "--data-folder pipeline_data "
        "--move-to-input "
        "--max-queries 500 "
        "--max-attempts 10 "
        "--parallel "
        "--max-queries-json max_queries_dict.json"
    )
    assert result.exit_code == 0

    assert not os.path.isfile("test-exp-not-in-input.jsonl")

    assert "No environment file found at some_file.env" in result.stderr
    assert (
        "Not creating judge file as one of judge_folder, judge or templates is None"
        in result.stderr
    )
    assert (
        "File test-exp-not-in-input.jsonl is not in the input folder pipeline_data/input"
        in result.stderr
    )
    assert (
        "Moving file from test-exp-not-in-input.jsonl to pipeline_data/input/test-exp-not-in-input.jsonl"
        in result.stderr
    )
    assert (
        "Settings: "
        "data_folder=pipeline_data, "
        "max_queries=500, "
        "max_attempts=10, "
        "parallel=True, "
        "max_queries_dict={'test': {'model1': 100, 'model2': 120}}\n"
        "Subfolders: "
        "input_folder=pipeline_data/input, "
        "output_folder=pipeline_data/output, "
        "media_folder=pipeline_data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: test-exp-not-in-input.jsonl..."
        in result.stderr
    )
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("pipeline_data/output/test-exp-not-in-input")


def test_run_experiment_judge_not_in_input_copy(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file test-exp-not-in-input.jsonl "
        "--data-folder pipeline_data "
        "--max-queries=200 "
        "--judge-folder judge_loc "
        "--judge judge1"
    )
    assert result.exit_code == 0

    assert os.path.isfile("test-exp-not-in-input.jsonl")

    assert "No environment file found at .env" in result.stderr
    assert "Judge folder loaded from judge_loc" in result.stderr
    assert "Templates to be used: ['template.txt']" in result.stderr
    assert "Judges to be used: ['judge1']" in result.stderr
    assert (
        "File test-exp-not-in-input.jsonl is not in the input folder pipeline_data/input"
        in result.stderr
    )
    assert (
        "Copying file from test-exp-not-in-input.jsonl to pipeline_data/input/test-exp-not-in-input.jsonl"
        in result.stderr
    )
    assert (
        "Settings: "
        "data_folder=pipeline_data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=pipeline_data/input, "
        "output_folder=pipeline_data/output, "
        "media_folder=pipeline_data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: test-exp-not-in-input.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: test-exp-not-in-input.jsonl" in result.stderr
    assert (
        "Starting processing judge of experiment: judge-test-exp-not-in-input.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: judge-test-exp-not-in-input.jsonl" in result.stderr
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("pipeline_data/output/test-exp-not-in-input")
    assert os.path.isdir("pipeline_data/output/judge-test-exp-not-in-input")


def test_run_experiment_judge(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200 "
        "--judge-folder judge_loc "
        "--judge-templates template.txt "
        "--judge judge1,judge2"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert "Judge folder loaded from judge_loc" in result.stderr
    assert "Templates to be used: ['template.txt']" in result.stderr
    assert "Judges to be used: ['judge1', 'judge2']" in result.stderr
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: test-experiment.jsonl" in result.stderr
    assert (
        "Starting processing judge of experiment: judge-test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: judge-test-experiment.jsonl" in result.stderr
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")
    assert os.path.isdir("data/output/judge-test-experiment")


def test_run_experiment_scorer_not_in_dict(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200 "
        "--scorer not_a_scorer"
    )
    assert result.exit_code != 0
    assert (
        "Scorer 'not_a_scorer' is not a key in scoring_functions_dict. "
        "Available scorers are: "
    ) in result.stderr


def test_run_experiment_scorer_only(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200 "
        "--scorer match,includes"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert (
        "Not creating judge file as one of judge_folder, judge or templates is None"
        in result.stderr
    )
    assert "Scoring functions to be used: ['match', 'includes']" in result.stderr
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.jsonl..."
        in result.stderr
    )
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")

    completed_files = [
        x for x in os.listdir("data/output/test-experiment") if "completed" in x
    ]
    assert len(completed_files) == 1
    completed_file = completed_files[0]

    # load the output to check the scores have been added
    with open(f"data/output/test-experiment/{completed_file}", "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    assert len(responses) == 2
    for response in responses:
        if response["id"] == 0:
            assert response["match"] is True
            assert response["includes"] is True
        elif response["id"] == 1:
            assert response["match"] is False
            assert response["includes"] is False
        else:
            assert False


def test_run_experiment_judge_and_scorer(temporary_data_folder_judge):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.jsonl "
        "--max-queries=200 "
        "--judge-folder judge_loc "
        "--judge-templates template.txt,template2.txt "
        "--judge judge2 "
        "--scorer 'match, includes'"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert "Judge folder loaded from judge_loc" in result.stderr
    assert "Templates to be used: ['template.txt', 'template2.txt']" in result.stderr
    assert "Judges to be used: ['judge2']" in result.stderr
    assert "Scoring functions to be used: ['match', 'includes']" in result.stderr
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: test-experiment.jsonl" in result.stderr
    assert (
        "Starting processing judge of experiment: judge-test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: judge-test-experiment.jsonl" in result.stderr
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")
    assert os.path.isdir("data/output/judge-test-experiment")

    # check the output files for the test-experiment
    completed_files = [
        x for x in os.listdir("data/output/test-experiment") if "completed" in x
    ]
    assert len(completed_files) == 1
    completed_file = completed_files[0]

    # load the output to check the scores have been added
    with open(f"data/output/test-experiment/{completed_file}", "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    # test that the scorers got added to the completed file
    assert len(responses) == 2
    for response in responses:
        if response["id"] == 0:
            assert response["match"] is True
            assert response["includes"] is True
        elif response["id"] == 1:
            assert response["match"] is False
            assert response["includes"] is False
        else:
            assert False

    # check the output files for the judge-test-experiment
    completed_files = [
        x for x in os.listdir("data/output/judge-test-experiment") if "completed" in x
    ]
    assert len(completed_files) == 1
    completed_file = completed_files[0]

    # load the output to check the scores have been added
    with open(f"data/output/judge-test-experiment/{completed_file}", "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    # test that the scorers got added to the completed judge file
    assert len(responses) == 4
    for response in responses:
        if response["id"] == "judge-judge2-template-0":
            assert response["input-match"] is True
            assert response["input-includes"] is True
        elif response["id"] == "judge-judge2-template-1":
            assert response["input-match"] is False
            assert response["input-includes"] is False
        elif response["id"] == "judge-judge2-template2-0":
            assert response["input-match"] is True
            assert response["input-includes"] is True
        elif response["id"] == "judge-judge2-template2-1":
            assert response["input-match"] is False
            assert response["input-includes"] is False
        else:
            assert False


def test_run_experiment_judge_and_scorer_with_csv_input_and_output(
    temporary_data_folder_judge,
):
    result = shell(
        "prompto_run_experiment "
        "--file data/input/test-experiment.csv "
        "--max-queries=200 "
        "--judge-folder judge_loc "
        "--judge-templates template.txt,template2.txt "
        "--judge judge2 "
        "--scorer 'match, includes' "
        "--output-as-csv"
    )
    assert result.exit_code == 0
    assert "No environment file found at .env" in result.stderr
    assert "Judge folder loaded from judge_loc" in result.stderr
    assert "Templates to be used: ['template.txt', 'template2.txt']" in result.stderr
    assert "Judges to be used: ['judge2']" in result.stderr
    assert "Scoring functions to be used: ['match', 'includes']" in result.stderr
    assert (
        "Settings: "
        "data_folder=data, "
        "max_queries=200, "
        "max_attempts=5, "
        "parallel=False\n"
        "Subfolders: "
        "input_folder=data/input, "
        "output_folder=data/output, "
        "media_folder=data/media"
    ) in result.stderr
    assert (
        "Starting processing experiment: data/input/test-experiment.csv..."
        in result.stderr
    )
    assert "Completed experiment: test-experiment.csv" in result.stderr
    assert (
        "Starting processing judge of experiment: judge-test-experiment.jsonl..."
        in result.stderr
    )
    assert "Completed experiment: judge-test-experiment.jsonl" in result.stderr
    assert "Experiment processed successfully!" in result.stderr
    assert os.path.isdir("data/output/test-experiment")
    assert os.path.isdir("data/output/judge-test-experiment")

    # check the output files for the test-experiment
    completed_files = [
        x for x in os.listdir("data/output/test-experiment") if "completed" in x
    ]
    # should be 2 (one jsonl and one csv)
    assert len(completed_files) == 2
    completed_jsonl_file = [
        file for file in completed_files if file.endswith(".jsonl")
    ][0]
    completed_csv_files = [file for file in completed_files if file.endswith(".csv")]
    assert len(completed_csv_files) == 1

    # load the output to check the scores have been added
    with open(f"data/output/test-experiment/{completed_jsonl_file}", "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    # test that the scorers got added to the completed file
    assert len(responses) == 2
    for response in responses:
        if response["id"] == 0:
            assert response["match"] is True
            assert response["includes"] is True
        elif response["id"] == 1:
            assert response["match"] is False
            assert response["includes"] is False
        else:
            assert False

    # check the output files for the judge-test-experiment
    completed_files = [
        x for x in os.listdir("data/output/judge-test-experiment") if "completed" in x
    ]
    # should be 2 (one jsonl and one csv)
    assert len(completed_files) == 2
    completed_jsonl_file = [
        file for file in completed_files if file.endswith(".jsonl")
    ][0]
    completed_csv_files = [file for file in completed_files if file.endswith(".csv")]
    assert len(completed_csv_files) == 1

    # load the output to check the scores have been added
    with open(f"data/output/judge-test-experiment/{completed_jsonl_file}", "r") as f:
        responses = [dict(json.loads(line)) for line in f]

    # test that the scorers got added to the completed judge file
    assert len(responses) == 4
    for response in responses:
        if response["id"] == "judge-judge2-template-0":
            assert response["input-match"] is True
            assert response["input-includes"] is True
        elif response["id"] == "judge-judge2-template-1":
            assert response["input-match"] is False
            assert response["input-includes"] is False
        elif response["id"] == "judge-judge2-template2-0":
            assert response["input-match"] is True
            assert response["input-includes"] is True
        elif response["id"] == "judge-judge2-template2-1":
            assert response["input-match"] is False
            assert response["input-includes"] is False
        else:
            assert False
