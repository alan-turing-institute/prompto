import argparse
import json
import os

from prompto.judge import Judge, load_judge_folder
from prompto.utils import parse_list_arg


def obtain_output_filepath(input_filepath: str, output_folder: str) -> str:
    input_filename = os.path.basename(input_filepath).replace("completed-", "")
    out_filepath = os.path.join(output_folder, f"judge-{input_filename}")
    return out_filepath


def main():
    """
    Generate a file for the judge-llm experiment using the responses from a completed file.
    """
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        "-i",
        help="Path to the input file containing the responses to judge",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--judge-folder",
        "-l",
        help=(
            "Location of the judge folder storing the template.txt "
            "and settings.json to be used"
        ),
        type=str,
        required=True,
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
        required=True,
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        help="Location where the judge file will be created",
        type=str,
        default="./",
    )
    args = parser.parse_args()

    # parse input file
    input_filepath = args.input_file
    try:
        with open(input_filepath, "r") as f:
            responses = [dict(json.loads(line)) for line in f]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Input file '{input_filepath}' is not a valid input file"
        ) from exc

    # parse template, judge folder and judge arguments
    templates = parse_list_arg(argument=args.templates)
    template_prompts, judge_settings = load_judge_folder(
        judge_folder=args.judge_folder, templates=templates
    )
    judge = parse_list_arg(argument=args.judge)
    # check if the judge is in the judge settings dictionary
    Judge.check_judge_in_judge_settings(judge=judge, judge_settings=judge_settings)

    # create output file path name
    out_filepath = obtain_output_filepath(
        input_filepath=input_filepath, output_folder=args.output_folder
    )

    # create judge object from the parsed arguments
    j = Judge(
        completed_responses=responses,
        judge_settings=judge_settings,
        template_prompts=template_prompts,
    )

    # create judge file
    j.create_judge_file(judge=judge, out_filepath=out_filepath)


if __name__ == "__main__":
    main()
