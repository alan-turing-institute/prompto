import argparse
import json
import logging
import os

from dotenv import load_dotenv

from prompto.experiment_pipeline import ExperimentPipeline
from prompto.settings import Settings


def main():
    """
    Constantly checks the input folder for new files
    and proccesses them sequentially (ordered by creation time).
    """
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        "-d",
        help="Path to the folder containing the data",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--env-file",
        "-e",
        help="Path to the environment file",
        type=str,
        default=".env",
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

    if args.max_queries_json is not None:
        # check if file exists
        if not os.path.exists(args.max_queries_json):
            raise FileNotFoundError(f"File {args.max_queries_json} not found")

        # check if file is a json file
        if not args.max_queries_json.endswith(".json"):
            raise ValueError("max_queries_json must be a json file")

        # load the json file
        with open(args.max_queries_json, "r") as f:
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
    logging.info(f"Starting to watch folder at {settings.input_folder}...")

    # initialise experiment pipeline
    experiment_pipeline = ExperimentPipeline(settings=settings)

    # run pipeline
    experiment_pipeline.run()


if __name__ == "__main__":
    main()
