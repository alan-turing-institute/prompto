import argparse
import os

from prompto.judge import Judge, parse_judge_arg, parse_judge_location_arg


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
        "--judge-location",
        "-l",
        help=(
            "Location of the judge folder storing the template.txt "
            "and settings.json to be used"
        ),
        type=str,
        required=True,
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

    # parse judge location and judge arguments
    template_prompt, judge_settings = parse_judge_location_arg(args.judge_location)
    judge = parse_judge_arg(args.judge)
    # check if the judge is in the judge settings dictionary
    Judge.check_judge_in_judge_settings(judge=judge, judge_settings=judge_settings)

    # parse input file
    input_filepath = args.input_file
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            responses = f.readlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Input file '{input_filepath}' does not exist"
        ) from exc

    # create output file path name
    input_filename = (
        os.path.basename(input_filepath)
        .removesuffix(".jsonl")
        .replace("completed-", "")
    )
    out_filepath = os.path.join(args.output_folder, f"judge-{input_filename}.jsonl")

    # create judge object from the parsed arguments
    judge = Judge(
        completed_responses=responses,
        judge_settings=judge_settings,
        template_prompt=template_prompt,
    )

    # create judge file
    judge.create_judge_file(judge=judge, out_filepath=out_filepath)


if __name__ == "__main__":
    main()
