import argparse
import json
import logging
import os
from datetime import datetime

from batch_llm.utils import write_log_message


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
    read_line_error_indices = []
    prompt_key_error_indices = []
    model_key_error_indices = []
    parameter_error_indices = []
    multimedia_error_indices = []
    multimedia_does_not_exist_indices = []
    multimedia_path_errors = []
    vertex_project_id_checked = False
    vertex_project_id_env_error = False
    llm_api_url_checked = False
    llm_api_url_env_error = False
    llm_api_error = False
    llm_api_exception = None
    gemini_api_endpoint_env_error = False
    gemini_project_id_env_error = False
    gemini_location_env_error = False
    gemini_model_id_env_error = False
    palm_2_checked = False
    gemini_checked = False
    gemini_model_indices = []
    safety_filter_error_indices = []
    valid_indicator = True

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            try:
                # check if line is a valid json
                data = json.loads(line)

                # check if "prompt" is a key in the json
                if "prompt" not in data:
                    # if "prompt" is not a key, add index to list
                    prompt_key_error_indices.append(i)
                    valid_indicator = False

                # if parameters is passed, check its a dictionary
                if "parameters" in data:
                    if type(data["parameters"]) is not dict:
                        parameter_error_indices.append(i)
                        valid_indicator = False

                # if multimedia is passed, check its a dictionary
                if "multimedia" in data:
                    if isinstance(data["multimedia"], dict):
                        data["multimedia"] = [data["multimedia"]]
                    for m in data["multimedia"]:
                        # each item must be a dictionary
                        if type(m) is not dict:
                            multimedia_error_indices.append(i)
                            valid_indicator = False

                        # must include "type" key
                        if "type" not in m:
                            multimedia_error_indices.append(i)
                            valid_indicator = False
                        else:
                            if m["type"] not in ["image", "video", "text"]:
                                multimedia_error_indices.append(i)
                                valid_indicator = False

                            if m["type"] in ["image", "video"]:
                                # check that the media file exists
                                if m.get("media") is not None:
                                    path_to_check = os.path.join(
                                        media_folder, m["media"]
                                    )
                                    if not os.path.exists(path_to_check):
                                        multimedia_does_not_exist_indices.append(i)
                                        multimedia_path_errors.append(path_to_check)
                                        valid_indicator = False

                        # must include "media" key
                        if "media" not in m:
                            multimedia_error_indices.append(i)
                            valid_indicator = False

                        # if type is video, must include "mime_type" key
                        if m["type"] == "video":
                            if "mime_type" not in m:
                                multimedia_error_indices.append(i)
                                valid_indicator = False

                # check if "model" is a key in the json
                if "model" not in data:
                    # if "model" is not a key, add index to list
                    model_key_error_indices.append(i)
                    valid_indicator = False
                else:
                    # only check the VERTEX_PROJECT_ID environment variable once
                    if not vertex_project_id_checked and data["model"] == "vertexai":
                        # check if VERTEX_PROJECT_ID environment variable is set
                        project_id = os.environ.get("VERTEX_PROJECT_ID")
                        if project_id is None:
                            vertex_project_id_env_error = True
                            valid_indicator = False

                    # only check the LLM_API_URL environment variable once
                    if not llm_api_url_checked and data["model"] == "llm_api":
                        # check if LLM_API_URL environment variable is set
                        api_url = os.environ.get("LLM_API_URL")
                        if api_url is None:
                            llm_api_url_env_error = True
                            valid_indicator = False

                        # check if LLM_API_URL is a valid URL
                        try:
                            import requests

                            # querying the root endpoint of the LLM API
                            requests.get(f"{api_url}/")
                        except Exception as err:
                            llm_api_exception = f"An unexpected error occurred: {type(err).__name__} - {err}"
                            llm_api_error = True
                            valid_indicator = False

                        # set the LLM API checked flag to True
                        llm_api_url_checked = True

                    # only check the Gemini environment variables once
                    if not gemini_checked and data["model"] == "gemini":
                        # check if GEMINI_MODEL_ID environment variable is set
                        model_id = os.environ.get("GEMINI_MODEL_ID")
                        if model_id is None:
                            gemini_model_id_env_error = True
                            valid_indicator = False

                        # set the Gemini checked flag to True
                        gemini_checked = True

                    if data["model"] == "gemini":
                        # must have safety filter setting
                        if "safety_filter" not in data or data["safety_filter"] not in [
                            "none",
                            "few",
                            "some",
                            "default",
                            "most",
                        ]:
                            safety_filter_error_indices.append(i)
                            valid_indicator = False
            except json.JSONDecodeError:
                # if line is not a valid json, add index to list
                read_line_error_indices.append(i)
                valid_indicator = False

    if not valid_indicator:
        log_msg = f"File {file_path} is an invalid jsonl file"
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)
    else:
        logging.info(f"File {file_path} is a valid jsonl file")

    if len(read_line_error_indices) != 0:
        log_msg = (
            f"File {file_path} has invalid json at line(s) {read_line_error_indices}"
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if len(prompt_key_error_indices) != 0:
        log_msg = f"File {file_path} has no 'prompt' key at line(s) {prompt_key_error_indices}"
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if len(model_key_error_indices) != 0:
        log_msg = (
            f"File {file_path} has no 'model' key at line(s) {model_key_error_indices}"
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if len(parameter_error_indices) != 0:
        log_msg = (
            f"File {file_path} has invalid 'parameters' key at line(s) {parameter_error_indices} "
            "- Make sure it is a dictionary of parameters."
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if len(multimedia_error_indices) != 0:
        log_msg = (
            f"File {file_path} has invalid 'multimedia' key at line(s) {multimedia_error_indices} "
            "- Make sure it is a dictionary and has keys 'type', 'media' (and 'mime' if type='video'). "
            " Also, make sure 'type' is one of 'image', 'video', 'text'."
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if len(multimedia_does_not_exist_indices) != 0:
        log_msg = (
            f"File {file_path} has invalid 'media' key at line(s) {multimedia_does_not_exist_indices} "
            "- Make sure the file exists."
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if len(multimedia_path_errors) != 0:
        multimedia_path_errors = set(multimedia_path_errors)
        log_msg = f"File {file_path} includes the following multimedia paths that do not exist: {multimedia_path_errors}"
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if vertex_project_id_env_error:
        log_msg = "Using model=='vertexai', but environment variable 'VERTEX_PROJECT_ID' not set"
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if llm_api_url_env_error:
        log_msg = (
            "Using model=='llm_api', but environment variable 'LLM_API_URL' not set"
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if llm_api_error:
        log_msg = (
            "Using model=='llm_api', but making a request to 'LLM_API_URL' "
            f"{api_url} failed with exception {llm_api_exception}"
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if gemini_api_endpoint_env_error:
        log_msg = "Using model=='gemini', but environment variable 'GEMINI_API_ENDPOINT' not set"
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if gemini_project_id_env_error:
        log_msg = "Using model=='gemini', but environment variable 'GEMINI_PROJECT_ID' not set"
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if gemini_location_env_error:
        log_msg = (
            "Using model=='gemini', but environment variable 'GEMINI_LOCATION' not set"
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if gemini_model_id_env_error:
        log_msg = (
            "Using model=='gemini', but environment variable 'GEMINI_MODEL_ID' not set"
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if len(gemini_model_indices) != 0:
        log_msg = (
            f"File {file_path} has model=='gemini' at line(s) {gemini_model_indices} "
            "- Make sure to use model='palm-2' or model='gemini' instead."
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

    if len(safety_filter_error_indices) != 0:
        log_msg = (
            f"File {file_path} has invalid 'safety_filter' key at line(s) {safety_filter_error_indices} "
            "- Make sure it is one of 'none', 'few', 'some', 'default', 'most'."
        )
        logging.info(log_msg)
        if log_file is not None:
            write_log_message(log_file=log_file, log_message=log_msg)

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
        help="Path to the folder containing the data.",
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

    # TODO: move file to input folder if valid if --move-to-input is used

    # log result
    if valid:
        logging.info(f"SUCCESS: {args.file} is valid.")
    else:
        logging.error(f"{args.file} is not valid. Please check logs and fix.")
        if args.log_file is not None:
            logging.error(f"See {args.log_file} for more details.")


if __name__ == "__main__":
    main()
