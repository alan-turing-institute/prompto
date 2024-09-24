import argparse
import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from prompto.experiment import Experiment
from prompto.judge import Judge, load_judge_folder
from prompto.scorer import SCORING_FUNCTIONS, obtain_scoring_functions
from prompto.settings import Settings
from prompto.utils import copy_file, move_file, parse_list_arg


def load_env_file(env_file: str) -> bool:
    """
    Load environment variables from a .env file using
    dotenv.load_dotenv.

    Will log info if the file is loaded successfully and
    a warning if the file is not found.

    Parameters
    ----------
    env_file : str
        Path to the .env file to load

    Returns
    -------
    bool
        Returned from dotenv.load_dotenv
    """
    loaded = load_dotenv(env_file)
    if loaded:
        logging.info(f"Loaded environment variables from {env_file}")
    else:
        logging.warning(f"No environment file found at {env_file}")

    return loaded


def load_max_queries_json(max_queries_json: str | None) -> dict:
    """
    Load the max queries json file if it is provided
    and returns as a dictionary.

    Raises errors if either the file does not exist
    or if it is not a json file.

    If the max_queries_json is None, an empty dictionary
    is returned.

    Parameters
    ----------
    max_queries_json : str | None
        Path to the json file containing the maximum queries
        per minute for each API and model or group as a dictionary.
        If None, an empty dictionary is returned

    Returns
    -------
    dict
        The dictionary containing the maximum queries per minute
        for each API and model or group
    """
    if max_queries_json is None:
        return {}

    # check if file exists
    if not os.path.exists(max_queries_json):
        raise FileNotFoundError(f"File {max_queries_json} not found")

    # check if file is a json file
    if not max_queries_json.endswith(".json"):
        raise ValueError("max_queries_json must be a json file")

    # load the json file
    with open(max_queries_json, "r") as f:
        max_queries_dict = json.load(f)

    return max_queries_dict


def load_judge_args(
    judge_folder_arg: str | None,
    judge_arg: str | None,
    templates_arg: str | None,
) -> tuple[bool, str, dict, list[str]]:
    """
    Load the judge arguments and parse them to get the
    template prompt, judge settings and judge.

    Also returns a boolean indicating if a judge file
    should be created and processed.

    Parameters
    ----------
    judge_folder_arg : str | None
        Path to judge folder containing the template.txt
        and settings.json files
    judge_arg : str | None
        Judge(s) to be used separated by commas. These must be keys
        in the judge settings dictionary

    Returns
    -------
    tuple[bool, str, dict, list[str]]
        A tuple containing the boolean indicating if a judge file
        should be created, the template prompt string, the judge
        settings dictionary and the judge list
    """
    if (
        judge_folder_arg is not None
        and judge_arg is not None
        and templates_arg is not None
    ):
        create_judge_file = True
        # parse template, judge folder and judge arguments
        templates = parse_list_arg(argument=templates_arg)
        template_prompts, judge_settings = load_judge_folder(
            judge_folder=judge_folder_arg, templates=templates
        )
        judge = parse_list_arg(argument=judge_arg)
        # check if the judge is in the judge settings dictionary
        Judge.check_judge_in_judge_settings(judge=judge, judge_settings=judge_settings)
        logging.info(f"Judge folder loaded from {judge_folder_arg}")
        logging.info(f"Templates to be used: {templates}")
        logging.info(f"Judges to be used: {judge}")
    else:
        logging.info(
            "Not creating judge file as one of judge_folder, judge or templates is None"
        )
        create_judge_file = False
        template_prompts, judge_settings, judge = None, None, None

    return create_judge_file, template_prompts, judge_settings, judge


def parse_file_path_and_check_in_input(
    file_path: str, settings: Settings, move_to_input: bool = False
) -> str:
    """
    Parse the file path to get the experiment file name.

    If the file is not in the input folder, it is either
    moved or copied there for processing depending on the
    move_to_input flag.

    Raises errors if either the file does not exist
    or if it is not a jsonl file.

    Parameters
    ----------
    file_path : str
        Path to the experiment file
    settings : Settings
        Settings object for the experiment which contains
        the input folder path
    move_to_input : bool, optional
        Flag to indicate if the file should be moved to the input
        folder. If False, the file is copied to the input folder.
        If the file is already in the input folder, this flag has
        no effect but the file will still be processed which would
        lead it to be moved to the output folder in the end.
        Default is False

    Returns
    -------
    str
        Experiment file name (without the full directories in the path)
    """
    # check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    # check if file is a jsonl or csv file
    if not file_path.endswith(".jsonl") and not file_path.endswith(".csv"):
        raise ValueError("Experiment file must be a jsonl or csv file")

    # get experiment file name (without the path)
    experiment_file_name = os.path.basename(file_path)

    # if the experiment file is not in the input folder, move it there
    if experiment_file_name not in os.listdir(settings.input_folder):
        logging.info(
            f"File {file_path} is not in the input folder {settings.input_folder}"
        )
        if move_to_input:
            move_file(
                source=file_path,
                destination=f"{settings.input_folder}/{experiment_file_name}",
            )
        else:
            copy_file(
                source=file_path,
                destination=f"{settings.input_folder}/{experiment_file_name}",
            )

    return experiment_file_name


def create_judge_experiment(
    create_judge_file: bool,
    experiment: Experiment,
    template_prompts: dict[str, str] | None,
    judge_settings: dict | None,
    judge: list[str] | str | None,
) -> Experiment | None:
    """
    Create a judge experiment if the create_judge_file flag is True.

    This experiment object should have been processed before,
    so that the completed responses are available.
    If the experiment has not been processed, an error is raised.

    Parameters
    ----------
    create_judge_file : bool
        Flag to indicate if a judge experiment should be created
    experiment : Experiment
        The experiment object to create the judge experiment from.
        This is used to obtain the list of completed responses
        and to create the judge experiment and file name.
    template_prompts : str | None
        The template prompt string to be used for the judge
    judge_settings : dict | None
        The judge settings dictionary to be used for the judge
    judge : list[str] | str | None
        The judge(s) to be used for the judge experiment. These
        must be keys in the judge settings dictionary

    Returns
    -------
    Experiment | None
        The judge experiment object if create_judge_file is True,
        otherwise None
    """
    if create_judge_file:
        # if completed_responses is empty, raise an error
        if experiment.completed_responses == []:
            raise ValueError(
                f"Cannot create judge file for experiment {experiment.experiment_name} "
                "as completed_responses is empty"
            )

        if not isinstance(template_prompts, dict):
            raise TypeError(
                "If create_judge_file is True, template_prompts must be a dictionary"
            )
        if not isinstance(judge_settings, dict):
            raise TypeError(
                "If create_judge_file is True, judge_settings must be a dictionary"
            )
        if not isinstance(judge, list) and not isinstance(judge, str):
            raise TypeError(
                "If create_judge_file is True, judge must be a list of strings or a string"
            )

        # create judge object from the parsed arguments
        j = Judge(
            completed_responses=experiment.completed_responses,
            judge_settings=judge_settings,
            template_prompts=template_prompts,
        )

        # create judge file
        judge_file_path = f"judge-{experiment.experiment_name}.jsonl"
        j.create_judge_file(
            judge=judge,
            out_filepath=f"{experiment.settings.input_folder}/{judge_file_path}",
        )

        # create Experiment object
        judge_experiment = Experiment(
            file_name=judge_file_path, settings=experiment.settings
        )
    else:
        judge_experiment = None

    return judge_experiment


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
    parser.add_argument(
        "--judge-folder",
        "-l",
        help=(
            "Location of the judge folder storing the template.txt "
            "and settings.json to be used"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--templates",
        "-t",
        help=(
            "Template file(s) to be used for the judge separated by commas. "
            "These must be .txt files in the judge folder. "
            "By default, the template file is 'template.txt'"
        ),
        type=str,
        default="template.txt",
    )
    parser.add_argument(
        "--judge",
        "-j",
        help=(
            "Judge(s) to be used separated by commas. "
            "These must be keys in the judge settings dictionary"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--scorer",
        "-s",
        help=(
            "Scorer(s) to be used separated by commas. "
            "These must be keys in the scorer settings dictionary"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-as-csv",
        help="Output the results as a csv file",
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

    # load environment variables
    load_env_file(args.env_file)

    # load the max queries json file
    max_queries_dict = load_max_queries_json(args.max_queries_json)

    # check if judge arguments are provided
    create_judge_file, template_prompts, judge_settings, judge = load_judge_args(
        judge_folder_arg=args.judge_folder,
        judge_arg=args.judge,
        templates_arg=args.templates,
    )

    # check if scorer is provided, and if it is in the SCORING_FUNCTIONS dictionary
    if args.scorer is not None:
        scoring_functions = obtain_scoring_functions(
            scorer=parse_list_arg(args.scorer), scoring_functions_dict=SCORING_FUNCTIONS
        )
    else:
        scoring_functions = None

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

    # parse the file path
    experiment_file_name = parse_file_path_and_check_in_input(
        file_path=args.file, settings=settings, move_to_input=args.move_to_input
    )

    # create Experiment object
    experiment = Experiment(file_name=experiment_file_name, settings=settings)

    # process the experiment
    logging.info(f"Starting processing experiment: {args.file}...")
    await experiment.process(evaluation_funcs=scoring_functions)

    if args.output_as_csv:
        experiment.save_completed_responses_to_csv()

    # create judge experiment
    judge_experiment = create_judge_experiment(
        create_judge_file=create_judge_file,
        experiment=experiment,
        template_prompts=template_prompts,
        judge_settings=judge_settings,
        judge=judge,
    )

    if judge_experiment is not None:
        # process the experiment
        logging.info(
            f"Starting processing judge of experiment: {judge_experiment.file_name}..."
        )
        await judge_experiment.process()

        if args.output_as_csv:
            judge_experiment.save_completed_responses_to_csv()

    logging.info("Experiment processed successfully!")


if __name__ == "__main__":
    asyncio.run(main())


def cli():
    asyncio.run(main())
