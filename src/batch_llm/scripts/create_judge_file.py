import argparse
import json
import os


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
        help="`Location of the judge template and settings to be used",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--judge", "-j", help="Judge to be used", type=str, default="gemini"
    )

    parser.add_argument(
        "--output-folder",
        "-o",
        help="Location where the judge file will be created.",
        type=str,
        default="./",
    )
    args = parser.parse_args()

    input_filepath = args.input_file
    judge_location = args.judge_location
    judge = args.judge
    output_folder = args.output_folder

    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            responses = f.readlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Input file '{input_filepath}' does not exist."
        ) from exc

    try:
        template_path = os.path.join(judge_location, "template.txt")
        with open(template_path, "r", encoding="utf-8") as f:
            template_prompt = f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Template file '{template_path}' does not exist."
        ) from exc

    try:
        judge__settings_path = os.path.join(judge_location, "settings.json")
        with open(judge__settings_path, "r", encoding="utf-8") as f:
            judge_settings = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Settings file '{judge__settings_path}' does not exist."
        ) from exc

    if judge not in judge_settings:
        raise ValueError(f"Judge '{judge}' not in judge settings.")

    settings = judge_settings[judge]

    input_filename = (
        os.path.basename(input_filepath).split(".")[0].replace("completed-", "")
    )
    out_filepath = output_folder + "judge:" + input_filename + ".jsonl"

    with open(out_filepath, "w", encoding="utf-8") as f:
        for _, response in enumerate(responses):
            response = json.loads(response)
            prompt_id = "judge-" + str(response["id"])
            prompt = response["prompt"]
            response = response["response"]
            judge_prompt = (
                f"{template_prompt}\nQUESTION: {prompt}\nRESPONSE: {response}\nCLASS:"
            )
            f.write(
                json.dumps(
                    {
                        "id": prompt_id,
                        "prompt": judge_prompt,
                        "model": judge,
                        "parameters": settings,
                    }
                )
            )
            f.write("\n")


if __name__ == "__main__":
    main()
