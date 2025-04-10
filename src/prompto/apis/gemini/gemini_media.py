import asyncio
import base64
import json
import logging
import os
import time

import google.generativeai as genai
import tqdm
from dotenv import load_dotenv

from prompto.utils import compute_sha256_base64

# initialise logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    datefmt=r"%Y-%m-%d %H:%M:%S",
    format="%(asctime)s [%(levelname)8s] %(message)s",
    level=logging.INFO,
)


def remote_file_hash_base64(remote_file):
    """
    Convert a remote file's SHA256 hash (stored as a hex-encoded UTF-8 bytes object)
    to a base64-encoded string.
    """
    hex_str = remote_file.sha256_hash.decode("utf-8")
    raw_bytes = bytes.fromhex(hex_str)
    return base64.b64encode(raw_bytes).decode("utf-8")


async def wait_for_processing(file_obj, poll_interval=1):
    """
    Poll until the file is no longer in the 'PROCESSING' state.
    Returns the updated file object.
    """
    while file_obj.state.name == "PROCESSING":
        await asyncio.sleep(poll_interval)
        file_obj = genai.get_file(file_obj.name)
    return file_obj


async def upload_single_file(local_file_path, already_uploaded_files):
    """
    Upload the file at 'file_path' if it hasn't been uploaded yet.
    If a file with the same SHA256 (base64-encoded) hash exists, returns its name.
    Otherwise, uploads the file, waits for it to be processed,
    and returns the new file's name. Raises a ValueError if processing fails.

    Parameters
    ----------
    file_path : str
        Path to the file to be uploaded.
    already_uploaded_files : dict[str, str]
        Dictionary mapping file hashes to filenames of already uploaded files.
    Returns
    -------
    A tuple containing:
        - The remote path and filename of the file.
        - The local path and filename of the file. (This is always the same as
          `local_file_path` parameter, but is a convenience for gathering the
          results later.)
    """
    local_hash = compute_sha256_base64(local_file_path)

    if local_hash in already_uploaded_files:
        logger.info(
            f"File '{local_file_path}' already uploaded as '{already_uploaded_files[local_hash]}'"
        )
        return already_uploaded_files[local_hash], local_file_path

    # Upload the file if it hasn't been found.
    # Use asyncio.to_thread to run the blocking upload_file function in a separate thread.
    logger.info(f"Uploading {local_file_path} to Gemini API")
    file_obj = await asyncio.to_thread(genai.upload_file, local_file_path)
    file_obj = await wait_for_processing(file_obj)

    if file_obj.state.name == "FAILED":
        err_msg = (
            f"Failure uploaded file '{file_obj.name}'. Error: {file_obj.error_message}"
        )
        raise ValueError(err_msg)
    # logger.info(
    #     f"Uploaded file '{file_obj.name}' with hash '{local_hash}' to Gemini API"
    # )
    already_uploaded_files[local_hash] = file_obj.name
    return file_obj.name, local_file_path


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
    logger.info(f"Found {len(uploaded_files)} files already uploaded at Gemini API")
    return uploaded_files


def list_uploaded_files():
    """
    List all previously uploaded files to the Gemini API.
    """
    _init_genai()
    uploaded_files = _get_previously_uploaded_files()

    for file_hash, file_name in uploaded_files.items():
        msg = f"File Name: {file_name}, File Hash: {file_hash}"
        logger.info(msg)
    logger.info("All uploaded files listed.")


def delete_uploaded_files():
    """
    Delete all previously uploaded files from the Gemini API.
    """
    _init_genai()
    uploaded_files = _get_previously_uploaded_files()
    return asyncio.run(_delete_uploaded_files_async(uploaded_files))


async def _delete_uploaded_files_async(uploaded_files):
    tasks = []
    for file_name in uploaded_files.values():
        logger.info(f"Preparing to delete file: {file_name}")
        tasks.append(asyncio.to_thread(genai.delete_file, file_name))

    await tqdm.asyncio.tqdm.gather(*tasks)
    logger.info("All uploaded files deleted.")


def upload_media_files(files_to_upload: set[str]):
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
    return asyncio.run(_upload_media_files_async(files_to_upload))


async def _upload_media_files_async(files_to_upload: set[str]):
    start_time = time.time()
    logger.info(f"Start retrieving previously uploaded files ")
    uploaded_files = _get_previously_uploaded_files()
    next_time = time.time()
    logger.info(f"Retrieved list of previously uploaded files")

    # Upload files asynchronously
    tasks = []
    for file_path in files_to_upload:
        logger.info(f"checking if {file_path} needs to be uploaded")
        tasks.append(upload_single_file(file_path, uploaded_files))

    remote_local_pairs = await tqdm.asyncio.tqdm.gather(*tasks)

    uploaded_files = {}
    for remote_filename, local_filename in remote_local_pairs:
        uploaded_files[local_filename] = remote_filename

    logger.info("All files uploaded.")
    return uploaded_files
