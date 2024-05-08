import argparse
import logging
import os

from prompto.experiment_processing import Experiment, ExperimentPipeline
from prompto.settings import Settings
from prompto.utils import move_file


def main():
    """
    Runs a particular experiment in the input data folder.
    """
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-file",
        "-e",
        help=(
            "Path to the experiment file. "
            "If it's not already in the input folder of the data folder provided, "
            "it is moved into the input folder. "
            "If it is already in the input folder, you can either "
            "provide the full path or just the filename."
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-folder",
        "-d",
        help="Path to the folder containing the data",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--max-queries",
        "-m",
        help="Maximum number of queries to send within a minute",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-attempts",
        "-a",
        help="Maximum number of attempts to process an experiment",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--parallel",
        "-p",
        help="Run the pipeline in parallel",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    # initialise logging
    logging.basicConfig(
        datefmt=r"%Y-%m-%d %H:%M:%S",
        format="%(asctime)s [%(levelname)8s] %(message)s",
        level=logging.INFO,
    )

    # initialise settings
    settings = Settings(
        data_folder=args.data_folder,
        max_queries=args.max_queries,
        max_attempts=args.max_attempts,
        parallel=args.parallel,
    )
    # log the settings that are set for the pipeline
    logging.info(settings)

    # check if file exists
    if not os.path.exists():
        raise FileNotFoundError(f"File {args.experiment_file} not found")
    if not args.experiment_file.endswith(".jsonl"):
        raise ValueError("Experiment file must be a jsonl file")

    # get experiment file name (without the path)
    file_name_split = args.experiment_file.split("/")
    experiment_file_name = file_name_split[-1]

    # if the experiment file is not in the input folder, move it there
    if experiment_file_name not in os.listdir(settings.input_folder):
        move_file(
            args.experiment_file, f"{settings.input_folder}/{experiment_file_name}"
        )

    # initialise experiment pipeline
    experiment_pipeline = ExperimentPipeline(settings=settings)

    # create Experiment object
    experiment = Experiment(file_name=args.experiment_file, settings=settings)

    # process the experiment
    experiment_pipeline.process_experiment(experiment=experiment)


if __name__ == "__main__":
    main()
