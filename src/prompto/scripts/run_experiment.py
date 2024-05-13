import argparse
import asyncio
import logging
import os

from prompto.experiment_processing import Experiment, ExperimentPipeline
from prompto.settings import Settings
from prompto.utils import move_file


async def main():
    """
    Runs a particular experiment in the input data folder.
    """
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        "-f",
        help=(
            "Path to the experiment file. "
            "If it's not already in the input folder of the data folder provided, "
            "it is moved into the input folder. "
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
        default=10,
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

    # get experiment file name (without the path)
    file_name_split = args.file.split("/")
    experiment_file_name = file_name_split[-1]

    # check if file exists or if it is in the input folder
    if not args.file.endswith(".jsonl"):
        raise ValueError("Experiment file must be a jsonl file")
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"File {args.file} not found")

    # if the experiment file is not in the input folder, move it there
    if experiment_file_name not in os.listdir(settings.input_folder):
        logging.info(
            f"File {args.file} is not in the input folder {settings.input_folder}"
        )
        move_file(args.file, f"{settings.input_folder}/{experiment_file_name}")

    # initialise experiment pipeline
    experiment_pipeline = ExperimentPipeline(settings=settings)

    # create Experiment object
    experiment = Experiment(file_name=experiment_file_name, settings=settings)

    # process the experiment
    logging.info(f"Processing experiment {experiment.experiment_name}...")
    await experiment_pipeline.process_experiment(experiment=experiment)


if __name__ == "__main__":
    asyncio.run(main())


def cli():
    asyncio.run(main())
