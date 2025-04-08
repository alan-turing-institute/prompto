import base64
import json
import os
import time

import google.generativeai as genai
import tqdm
from dotenv import load_dotenv

from prompto.utils import compute_sha256_base64

# TODO: replace print with logging


def remote_file_hash_base64(remote_file):
    """
    Convert a remote file's SHA256 hash (stored as a hex-encoded UTF-8 bytes object)
    to a base64-encoded string.
    """
    hex_str = remote_file.sha256_hash.decode("utf-8")
    raw_bytes = bytes.fromhex(hex_str)
    return base64.b64encode(raw_bytes).decode("utf-8")


def wait_for_processing(file_obj, poll_interval=10):
    """
    Poll until the file is no longer in the 'PROCESSING' state.
    Returns the updated file object.
    """
    while file_obj.state.name == "PROCESSING":
        print("Waiting for file to be processed...")
        time.sleep(poll_interval)
        file_obj = genai.get_file(file_obj.name)
    return file_obj


def upload(file_path, already_uploaded_files):
    """
    Upload the file at 'file_path' if it hasn't been uploaded yet.
    If a file with the same SHA256 (base64-encoded) hash exists, returns its name.
    Otherwise, uploads the file, waits for it to be processed,
    and returns the new file's name. Raises a ValueError if processing fails.
    """
    local_hash = compute_sha256_base64(file_path)

    if local_hash in already_uploaded_files:
        return already_uploaded_files[local_hash], already_uploaded_files

    # Upload the file if it hasn't been found.
    file_obj = genai.upload_file(path=file_path)
    file_obj = wait_for_processing(file_obj)

    if file_obj.state.name == "FAILED":
        raise ValueError("File processing failed")
    already_uploaded_files[local_hash] = file_obj.name
    return already_uploaded_files[local_hash], already_uploaded_files


def _init_genai():
    load_dotenv(dotenv_path=".env")
    # TODO: check if this can be refactored to a common function
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY is None:
        raise ValueError("GEMINI_API_KEY is not set")

    genai.configure(api_key=GEMINI_API_KEY)


def _get_previously_uploaded_files():
    uploaded_files = {
        remote_file_hash_base64(remote_file): remote_file.name
        for remote_file in genai.list_files()
    }
    print(f"Found {len(uploaded_files)} files already uploaded")
    return uploaded_files


def list_uploaded_files():
    """
    List all previously uploaded files to the Gemini API.
    """
    _init_genai()
    uploaded_files = _get_previously_uploaded_files()

    for file_hash, file_name in uploaded_files.items():
        print(f"File Hash: {file_hash}, File Name: {file_name}")
    print("All uploaded files listed.")


def delete_uploaded_files():
    """
    Delete all previously uploaded files from the Gemini API.
    """
    _init_genai()
    uploaded_files = _get_previously_uploaded_files()

    for file_name in uploaded_files.values():
        print(f"Deleting file: {file_name}")
        genai.delete_file(file_name)
    print("All uploaded files deleted.")


def upload_media(files_to_upload: set[str]):
    """
    Upload media files to the Gemini API.

    Parameters:
    ----------
    files_to_upload : set[str]
        Set of absolute, local, paths of files to upload.

    Returns:
    -------
    dict[str, str]
        Dictionary mapping local file paths to their corresponding uploaded filenames.
    """
    _init_genai()
    uploaded_files = _get_previously_uploaded_files()

    # Upload files and store mappings
    genai_files = {}
    for file_path in tqdm.tqdm(files_to_upload):
        print(f"Uploading {file_path}")
        uploaded_filename, uploaded_files = upload(file_path, uploaded_files)
        genai_files[file_path] = uploaded_filename
        print(f"Uploaded {file_path} as {uploaded_filename}")

    print("All files uploaded.")

    return genai_files
