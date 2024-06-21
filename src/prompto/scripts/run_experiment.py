import argparse
import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from prompto.experiment import Experiment
from prompto.settings import Settings
from prompto.utils import copy_file, move_file


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
        "--env-file",
        "-e",
        help="Path to the environment file",
        type=str,
        default=".env",
    )
    parser.add_argument(
        "--move-to-input",
        "-m",
        help=(
            "If used, the file will be moved to the input folder to run. "
            "By default the file is only copied to the input folder. "
            "Note if the file is already in the input folder, this flag has no effect "
            "but the file will still be processed which would lead it to be "
            "moved to the output folder."
        ),
        action="store_true",
        default=False,
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
        "-mq",
        help="The default maximum number of queries to send per minute",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-attempts",
        "-ma",
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
    parser.add_argument(
        "--max-queries-json",
        "-mqj",
        help=(
            "Path to the json file containing the maximum queries per minute "
            "for each API and model or group"
        ),
        type=str,
        default=None,
    )
    args = parser.parse_args()

    # initialise logging
    logging.basicConfig(
        datefmt=r"%Y-%m-%d %H:%M:%S",
        format="%(asctime)s [%(levelname)8s] %(message)s",
        level=logging.INFO,
    )

    # load environment variables
    loaded = load_dotenv(args.env_file)
    if loaded:
        logging.info(f"Loaded environment variables from {args.env_file}")
    else:
        logging.warning(f"No environment file found at {args.env_file}")

    if args.max_query_json is not None:
        # check if file exists
        if not os.path.exists(args.max_query_json):
            raise FileNotFoundError(f"File {args.max_query_json} not found")

        # check if file is a json file
        if not args.max_query_json.endswith(".json"):
            raise ValueError("max_query_json must be a json file")

        # load the json file
        with open(args.max_query_json, "r") as f:
            max_queries_dict = json.load(f)
    else:
        max_queries_dict = {}

    # initialise settings
    settings = Settings(
        data_folder=args.data_folder,
        max_queries=args.max_queries,
        max_attempts=args.max_attempts,
        parallel=args.parallel,
        max_queries_dict=max_queries_dict,
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
        if args.move_to_input:
            move_file(
                source=args.file,
                destination=f"{settings.input_folder}/{experiment_file_name}",
            )
        else:
            copy_file(
                source=args.file,
                destination=f"{settings.input_folder}/{experiment_file_name}",
            )

    # create Experiment object
    experiment = Experiment(file_name=experiment_file_name, settings=settings)

    # process the experiment
    logging.info(f"Processing experiment {experiment.experiment_name}...")
    await experiment.process()


if __name__ == "__main__":
    asyncio.run(main())


def cli():
    asyncio.run(main())
