import argparse
import logging

from batch_llm.experiment_processing import Experiment
from batch_llm.settings import Settings


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
        help="Path to the folder containing the data.",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--max-queries",
        "-m",
        help="Maximum number of queries to send within a minute.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--max-attempts",
        "-a",
        help="Maximum number of attempts to process an experiment.",
        type=int,
        default=5,
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
    )

    logging.info(f"Starting to watch folder at {settings.input_folder}...")

    # initialise experiment pipeline
    experiment_pipeline = ExperimentPipeline(settings=settings)

    # run pipeline
    experiment_pipeline.run()


if __name__ == "__main__":
    main()
