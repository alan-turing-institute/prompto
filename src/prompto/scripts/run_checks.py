import argparse
import json
import logging
import os

from prompto.models import ASYNC_MODELS
from prompto.utils import move_file, write_log_message


def check_multimedia(
    multimedia: dict | list, media_folder: str
) -> tuple[list[Exception], set[str]]:
    issues = []
    multimedia_path_errors = set()
    if isinstance(multimedia, dict):
        multimedia = [multimedia]

    if type(multimedia) is not list:
        issues.append(
            TypeError(
                '"multimedia" value must be a dictionary or a list of dictionaries if provided'
            )
        )

    for m in multimedia:
        # each item must be a dictionary
        if type(m) is not dict:
            issues.append(
                TypeError('Each item in "multimedia" value must be a dictionary')
            )

        # must include "type" key
        if "type" not in m:
            issues.append(
                KeyError(
                    'For each "multimedia" item provided, a "type" key must be provided'
                )
            )
        else:
            if m["type"] not in ["image", "video", "text"]:
                issues.append(
                    ValueError(
                        'For each "multimedia" item provided, "type" value must be '
                        "one of 'image', 'video', 'text'"
                    )
                )

            if m["type"] in ["image", "video"]:
                # check that the media file exists
                if m.get("media") is not None:
                    path_to_check = os.path.join(media_folder, m["media"])
                    if not os.path.exists(path_to_check):
                        issues.append(
                            FileNotFoundError(f"File '{path_to_check}' does not exist")
                        )
                        multimedia_path_errors.add(path_to_check)

        # must include "media" key
        if "media" not in m:
            issues.append(
                KeyError(
                    'For each "multimedia" item provided, a "media" key must be provided'
                )
            )

        # if type is video, must include "mime_type" key
        if m["type"] == "video":
            if "mime_type" not in m:
                issues.append(
                    KeyError(
                        "For each \"multimedia\" item provided with type 'video', "
                        'a "mime_type" key must be provided'
                    )
                )

    return issues, multimedia_path_errors


def is_valid_jsonl(
    file_path: str, media_folder: str, log_file: str | None = None
) -> bool:
    """
    Check if a file is a valid jsonl file and can be read line by line
    and if "prompt" is a key in all lines of the file.

    Parameters
    ----------
    file_path : str
        Path to the jsonl file to be checked.
    media_folder : str
        String containing the path to the media folder to be used.
    log_file : str | None
        Path to the error log file.
        Only used if the file is deemed invalid.
        Log will include the errors that caused the file to fail validation
        and the line numbers of the errors.
        If None, no error log file will be created. Default is None.

    Returns
    -------
    bool
        True if the file is a valid jsonl file, False otherwise.
    """
    multimedia_path_errors = set()
    valid_indicator = True
    if log_file is None:
        log_file = os.path.basename(file_path).replace(".jsonl", "_error_log.txt")
        logging.info("Log file not provided. Generating one in current directory")

    logging.info(
        f"Checking {file_path}. Any errors will be saved to log file at {log_file}"
    )

    if log_file is not None:
        with open(log_file, "a") as log:
            log.write("\n")
        write_log_message(log_file=log_file, log_message="Running checks...", log=True)

    model_environments_to_check = set()
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            issues = []
            try:
                # check if line is a valid json
                data = json.loads(line)

                # check if "prompt" is a key in the json
                if "prompt" not in data:
                    # if "prompt" is not a key, add index to list
                    issues.append(KeyError('"prompt" key not found'))

                # check if "api" is a key in the json
                if "api" not in data:
                    # if "api" is not a key, add index to list
                    issues.append(KeyError('"api" key not found'))

                # if parameters is passed, check its a dictionary
                if "parameters" in data:
                    if type(data["parameters"]) is not dict:
                        issues.append(
                            TypeError(
                                '"parameters" value must be a dictionary if provided'
                            )
                        )

                # if multimedia is passed, check its a dictionary
                if "multimedia" in data:
                    multimedia_issues, path_errors = check_multimedia(
                        data["multimedia"], media_folder
                    )
                    issues.extend(multimedia_issues)
                    multimedia_path_errors.add(path_errors)

                if "api" in data:
                    if data["api"] not in ASYNC_MODELS:
                        issues.append(
                            NotImplementedError(
                                f"Model {data['api']} is not a valid model. "
                                f"Please check the model name"
                            )
                        )
                    else:
                        # model specific checks
                        issues.extend(ASYNC_MODELS[data["api"]].check_prompt_dict(data))
                        # add model to set of models to check environment variables for
                        model_environments_to_check.add(data["api"])
            except json.JSONDecodeError as err:
                # if line is not a valid json, add index to list
                issues.append(err)

            if len(issues) != 0:
                if not all(isinstance(item, Warning) for item in issues):
                    valid_indicator = False
                # log the issues
                log_msg = f"Line {i} has the following issues: {issues}"
                if log_file is not None:
                    write_log_message(log_file=log_file, log_message=log_msg, log=True)

    # check environment variables for each model
    environment_issues = []
    for model in model_environments_to_check:
        environment_issues.extend(ASYNC_MODELS[model].check_environment_variables())

    if len(environment_issues) != 0:
        if not all(isinstance(item, Warning) for item in environment_issues):
            valid_indicator = False
        log_msg = (
            f"File {file_path} has the following environment variables "
            f"that aren't set: {environment_issues}"
        )
        write_log_message(log_file=log_file, log_message=log_msg)

    if len(multimedia_path_errors) != 0:
        valid_indicator = False
        log_msg = (
            f"File {file_path} includes the following multimedia paths "
            f"that do not exist: {multimedia_path_errors}"
        )
        write_log_message(log_file=log_file, log_message=log_msg)

    if not valid_indicator:
        log_msg = f"File {file_path} is an invalid jsonl file"
        write_log_message(log_file=log_file, log_message=log_msg)
    else:
        logging.info(
            f"File {file_path} is a valid jsonl file. But check if there's any warnings in the logs"
        )

    return valid_indicator


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        "-f",
        help="File to be checked",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--log-file",
        "-l",
        help="Log file to be written to",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-folder",
        "-d",
        help="Path to the folder containing the data",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--move-to-input",
        "-m",
        help="If used, the file will be moved to the input folder if it is valid",
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

    # check if file is valid
    valid = is_valid_jsonl(
        file_path=args.file,
        media_folder=f"{args.data_folder}/media",
        log_file=args.log_file,
    )

    # log result
    if valid:
        logging.info(f"SUCCESS: {args.file} is valid")
        if args.move_to_input:
            new_file_path = f"{args.data_folder}/input/{os.path.basename(args.file)}"
            move_file(args.file, new_file_path)
            logging.info(f"Moved {args.file} to {new_file_path}")
    else:
        logging.error(f"{args.file} is not valid. Please check logs and fix")
        if args.log_file is not None:
            logging.error(f"See {args.log_file} for more details")


if __name__ == "__main__":
    main()
