import argparse
import json
import logging
import os

import prompto.apis.gemini.gemini_media as gemini_media
from prompto.apis import ASYNC_APIS

# initialise logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    datefmt=r"%Y-%m-%d %H:%M:%S",
    format="%(asctime)s [%(levelname)8s] %(message)s",
    level=logging.INFO,
)


UPLOAD_APIS = {
    "gemini": gemini_media.upload_media_files,
}


def check_uploads_by_api(api_name: str):
    """
    Check if preemptive uploading media files has been implemented for the given API.
    """
    if api_name not in UPLOAD_APIS.keys():
        raise NotImplementedError(
            f"Uploading media files to {api_name} is not supported yet."
        )

    return True


def _read_experiment_file(
    experiment_path: str, media_dir: str
) -> tuple[set[str], list[dict]]:
    """
    Read the experiment file and collect all media file paths.
    Creates a set of absolute paths to the media files to be uploaded.

    Returns
    -------
    set[str]
        A set of absolute paths to the media files to be uploaded.
    list[dict]
        A list of prompt_dicts (prompt dictionaries) containing the data from
        an individual line in the experiment file.
    """

    # Read and collect media file paths
    with open(experiment_path, "r") as f:
        lines = f.readlines()

    prompt_dict_list = []
    files_to_upload: set[str] = set()

    for line in lines:
        data = json.loads(line)
        prompt_dict_list.append(data)

        if not isinstance(data.get("prompt"), list):
            continue

        absolute_media_paths = {
            os.path.join(media_dir, el["media"])
            for prompt in data["prompt"]
            for part in prompt.get("parts", [])
            if isinstance(el := part, dict) and "media" in el
        }

        if len(absolute_media_paths) > 0:
            # We only need to check the API, if we already know that there is a media file to upload
            check_uploads_by_api(data.get("api"))
            files_to_upload.update(absolute_media_paths)

    return files_to_upload, prompt_dict_list


def _resolve_output_file_location(args: argparse.Namespace):
    """
    Check if the output file location is valid.
    """
    input_file_name = os.path.splitext(args.file)[0]
    if args.output_file is None:
        # Output file location is not specified
        # Default to something based on the input file
        args.output_file = os.path.join(f"{input_file_name}_uploaded.jsonl")

    if not args.overwrite_output and os.path.exists(args.output_file):
        # Output file already exists
        # Raise an error if the user does not want to overwrite
        raise ValueError(
            f"Output file {args.output_file} already exists."
            " Use `--overwrite` to overwrite it, or `--output-file` to"
            " specify a different output file."
        )


def update_experiment_file(
    prompt_dict_list: list[dict],
    uploaded_files: dict[str, str],
    output_path: str,
    media_location: str,
) -> None:
    """
    Creates or updates the experiment file with the uploaded filenames.
    The uploaded filenames are added to the prompt dictionaries.

    Parameters:
    ----------
    prompt_dict_list : list[dict]
        A list of prompt dictionaries containing the data from the original experiment file.
    uploaded_files : dict[str, str]
        A dictionary mapping local file paths to their corresponding uploaded filenames.
    output_path : str
        The path for the new/updated experiment file. No checking of the
        overwrite behaviour is included in this function. It is assumed that
        the overwrite logic has been implemented elsewhere.
    media_location : str
        The location of the media files (e.g., "data/media").
    """
    # Modify data to include uploaded filenames
    for data in prompt_dict_list:
        if isinstance(data.get("prompt"), list):
            for prompt in data["prompt"]:
                for part in prompt.get("parts", []):
                    if isinstance(part, dict) and "media" in part:
                        file_path = f'{media_location}/{part["media"]}'
                        if file_path in uploaded_files:
                            part["uploaded_filename"] = uploaded_files[file_path]
                        else:
                            logger.warning(
                                f"Failed to find {file_path} in uploaded_files"
                            )

    # Write modified data back to the JSONL file
    with open(output_path, "w") as f:
        for data in prompt_dict_list:
            f.write(json.dumps(data) + "\n")


def upload_media_parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    upload_grp = subparsers.add_parser(
        name="upload",
        help="Upload media files to the relevant API. The media files are uploaded and the experiment file is updated with the uploaded filenames.",
    )
    upload_grp.add_argument(
        "--file",
        "-f",
        help=(
            "Path to the experiment file."
            " This file is not moved by the `prompto_upload_media upload` command"
        ),
        type=str,
        required=True,
    )
    upload_grp.add_argument(
        "--data-folder",
        "-d",
        help="Path to the folder containing the media files",
        type=str,
        default="data",
        required=True,
    )
    upload_grp.add_argument(
        "--output-file",
        "-o",
        help="Path to new or updated output file. A updated version of the input file is created with the path to the media files updated. If `--output-file` is specified, this value will be used. If `--output-file` is not specified, a new file will be created with the same name as the input file, but with `_uploaded` appended to the name. The input file can be overwritten if both the `--overwrite-output` option is set and the `--output-file` specifies the same path as `--file`.",
        type=str,
        default=None,
        required=False,
    )
    upload_grp.add_argument(
        "--overwrite-output",
        "-w",
        help="Overwrite the output file (if it exist). If this is not specified the command will refuse to overwrite the output file if it already exists.",
        action="store_true",
        default=False,
    )
    upload_grp.set_defaults(func=do_upload_media)

    delete_grp = subparsers.add_parser(
        name="delete",
        help="Delete previously uploaded files. No files will be uploaded.",
    )
    delete_grp.add_argument(
        "--confirm-delete-all",
        help="Delete existing files. This option is required to confirm that you want to delete all previously uploaded files.",
        action="store_true",
        default=False,
        required=True,
    )
    delete_grp.set_defaults(func=do_delete_existing_files)

    list_grp = subparsers.add_parser(
        name="list",
        help="List previously uploaded files.",
    )
    list_grp.set_defaults(func=do_list_uploaded_files)

    args = parser.parse_args()
    return args


def do_delete_existing_files(args):
    gemini_media.delete_uploaded_files()
    return


def do_list_uploaded_files(args):
    gemini_media.list_uploaded_files()
    return


def do_upload_media(args):
    _resolve_output_file_location(args)

    input_file = args.file
    data_folder = args.data_folder
    output_file = args.output_file

    files_to_upload, prompt_dict_list = _read_experiment_file(input_file, data_folder)

    # At present we only support the gemini API
    # Therefore we will just call the upload function
    # If in future we support other bulk upload to other APIs, we will need to
    # refactor here
    uploaded_files = gemini_media.upload_media_files(files_to_upload)

    update_experiment_file(
        prompt_dict_list,
        uploaded_files,
        output_file,
        data_folder,
    )


def main():
    args = upload_media_parse_args()
    args.func(args)


# Only intended for testing purposes only
if __name__ == "__main__":
    main()
