import asyncio
import base64
import logging

import tqdm
from google import genai

from prompto.apis.gemini.gemini import GeminiAPI
from prompto.settings import Settings
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
    hex_str = remote_file.sha256_hash
    raw_bytes = bytes.fromhex(hex_str)
    return base64.b64encode(raw_bytes).decode("utf-8")


async def wait_for_processing(file_obj, client: genai.Client, poll_interval=1):
    """
    Poll until the file is no longer in the 'PROCESSING' state.
    Returns the updated file object.
    """
    while file_obj.state.name == "PROCESSING":
        await asyncio.sleep(poll_interval)
        # We need to re-fetch the file object to get the updated state.
        file_obj = client.files.get(name=file_obj.name)
    return file_obj


async def _upload_single_file(
    local_file_path, already_uploaded_files, client: genai.Client
):
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
    logger.info(f"Uploading {local_file_path} to Gemini API")

    file_obj = await client.aio.files.upload(file=local_file_path)
    file_obj = await wait_for_processing(file_obj, client=client)

    if file_obj.state.name == "FAILED":
        err_msg = (
            f"Failure uploaded file '{file_obj.name}'. Error: {file_obj.error_message}"
        )
        raise ValueError(err_msg)

    logger.info(
        f"Uploaded file '{file_obj.name}' with hash '{local_hash}' to Gemini API"
    )
    already_uploaded_files[local_hash] = file_obj.name
    return file_obj.name, local_file_path


async def _get_previously_uploaded_files(client: genai.Client):
    raw_files = await client.aio.files.list()
    uploaded_files = {
        remote_file.sha256_hash: remote_file.name for remote_file in raw_files
    }
    logger.info(f"Found {len(uploaded_files)} files already uploaded at Gemini API")
    return uploaded_files


def list_uploaded_files(settings: Settings):
    """
    List all previously uploaded files to the Gemini API.
    """
    gemini_api = GeminiAPI(settings=settings, log_file=None)
    # TODO: We need a model name, because our API caters for different API keys
    # for different models. Maybe our API is too complicated....
    default_model_name = "default"
    client = gemini_api._get_client(default_model_name)
    uploaded_files = asyncio.run(_get_previously_uploaded_files(client))

    for file_hash, file_name in uploaded_files.items():
        msg = f"File Name: {file_name}, File Hash: {file_hash}"
        logger.info(msg)
    logger.info("All uploaded files listed.")


def delete_uploaded_files(settings: Settings):
    """
    Delete all previously uploaded files from the Gemini API.
    """
    gemini_api = GeminiAPI(settings=settings, log_file=None)
    # TODO: We need a model name, because our API caters for different API keys
    # for different models. Maybe our API to complicated....
    default_model_name = "default"
    client = gemini_api._get_client(default_model_name)

    # This just using the synchronous API. Using the async API did not
    # seem reliable. In particular `client.aio.files.delete()` did not appear
    # to always actually deleting the files (even after repeatedly polling the file)
    # This is not an important function in prompto and delete action is reasonably
    # quick, so we can live with this simple solution.
    # ``
    for remote_file in client.files.list():
        client.files.delete(name=remote_file.name)

    logger.info("All uploaded files deleted.")


def upload_media_files(files_to_upload: set[str], settings: Settings):
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
    # _init_genai()
    return asyncio.run(upload_media_files_async(files_to_upload, settings))


async def upload_media_files_async(files_to_upload: set[str], settings: Settings):
    logger.info("Start retrieving previously uploaded files")
    gemini_api = GeminiAPI(settings=settings, log_file=None)
    client = gemini_api._get_client("default")

    uploaded_files = await _get_previously_uploaded_files(client)

    logger.info("Retrieved list of previously uploaded files")

    # Upload files asynchronously
    tasks = []
    for file_path in files_to_upload:
        logger.info(f"checking if {file_path} needs to be uploaded")
        tasks.append(_upload_single_file(file_path, uploaded_files, client))

    remote_local_pairs = await tqdm.asyncio.tqdm.gather(*tasks)

    uploaded_files = {}
    for remote_filename, local_filename in remote_local_pairs:
        uploaded_files[local_filename] = remote_filename

    logger.info("All files uploaded.")
    return uploaded_files
