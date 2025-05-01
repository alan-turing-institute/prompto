import asyncio
import base64
import logging
import os
import tempfile
from time import sleep

import tqdm
from dotenv import load_dotenv
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
    # hex_str = remote_file.sha256_hash.decode("utf-8")
    hex_str = remote_file.sha256_hash
    raw_bytes = bytes.fromhex(hex_str)
    return base64.b64encode(raw_bytes).decode("utf-8")


async def wait_for_processing(file_obj, client: genai.Client, poll_interval=1):
    """
    Poll until the file is no longer in the 'PROCESSING' state.
    Returns the updated file object.
    """
    # print(f"File {file_obj.name} is in state {file_obj.state.name}")

    while file_obj.state.name == "PROCESSING":
        await asyncio.sleep(poll_interval)
        # We need to re-fetch the file object to get the updated state.
        file_obj = client.files.get(name=file_obj.name)
        # print(f"File {file_obj.name} is in state {file_obj.state.name}")
        # print(f"{file_obj.error=}")
        # print(f"{file_obj.update_time=}")
        # print(f"{file_obj.create_time=}")
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
    print(f"local_file_path: {local_file_path}")
    print(f"local_hash: {local_hash}")

    if local_hash in already_uploaded_files:
        logger.info(
            f"File '{local_file_path}' already uploaded as '{already_uploaded_files[local_hash]}'"
        )
        return already_uploaded_files[local_hash], local_file_path

    # Upload the file if it hasn't been found.
    # Use asyncio.to_thread to run the blocking upload_file function in a separate thread.
    logger.info(f"Uploading {local_file_path} to Gemini API")

    # file_obj = await asyncio.to_thread(genai.upload_file, local_file_path)
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


# def _init_genai():
#     load_dotenv(dotenv_path=".env")
#     # TODO: check if this can be refactored to a common function
#     GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
#     if GEMINI_API_KEY is None:
#         raise ValueError("GEMINI_API_KEY is not set")

#     genai.configure(api_key=GEMINI_API_KEY)


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
    # _init_genai()

    # Settings are not used in this function, but we need to
    # create a dummy settings object to pass to the GeminiAPI
    # TODO:
    #  Also, we don't need a directory, but Settings constructor
    # insists on creating these directories locally.
    # A better solution would be to create an option in the
    # Settings constructor to not create the directories.
    # But for now we'll just pass it a temporary directory.
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     data_folder = os.path.join(temp_dir, "data")
    #     os.makedirs(data_folder, exist_ok=True)
    #     dummy_settings = Settings(data_folder=data_folder)

    genmini_api = GeminiAPI(settings=settings, log_file=None)
    # TODO: We need a model name, because our API caters for different API keys
    # for different models. Maybe our API to complicated....
    default_model_name = "default"
    client = genmini_api._get_client(default_model_name)
    uploaded_files = asyncio.run(_get_previously_uploaded_files(client))

    for file_hash, file_name in uploaded_files.items():
        msg = f"File Name: {file_name}, File Hash: {file_hash}"
        logger.info(msg)
    logger.info("All uploaded files listed.")


def delete_uploaded_files(settings: Settings):
    """
    Delete all previously uploaded files from the Gemini API.
    """
    # _init_genai()

    # with tempfile.TemporaryDirectory() as temp_dir:
    #     data_folder = os.path.join(temp_dir, "data")
    #     os.makedirs(data_folder, exist_ok=True)
    #     dummy_settings = Settings(data_folder=data_folder)

    genmini_api = GeminiAPI(settings=settings, log_file=None)
    # TODO: We need a model name, because our API caters for different API keys
    # for different models. Maybe our API to complicated....
    default_model_name = "default"
    client = genmini_api._get_client(default_model_name)

    # uploaded_files = asyncio.run(_get_previously_uploaded_files(client))
    for remote_file in client.files.list():
        # file_name = file_name.name
        client.files.delete(name=remote_file.name)
        # _delete_single_uploaded_file(file_name, client)
    # return asyncio.run(_delete_uploaded_files_async(uploaded_files, client))
    logger.info("All uploaded files deleted.")


def _delete_single_uploaded_file(file_name: str, client: genai.Client):
    """
    Delete a single uploaded file from the Gemini API.
    """
    print(f"Deleting file {file_name}")
    file = client.files.get(name=file_name)
    client.files.delete(name=file_name)
    # indx = 0

    # The delete function is non-blocking (even the sync version)
    # and returns immediately. So we need to poll the file object
    # to see if it is still exists.
    # The only reliable way to check if the file is deleted is to
    # try and get it again and see if it raises an error.
    while True:
        # We need to re-fetch the file object to get the updated state.
        try:
            file = client.files.get(name=file.name)
            print(f"File {file.name} is in state {file.state.name}")
            print(f"{file.error=}")
            print(f"{file.update_time=}")
            print(f"{file.create_time=}")
            print(f"{indx=}")
            # client.files.delete(name=file_name)
            indx += 1
            # if indx > 10:
            #     break
        except genai.errors.ClientError as e:
            # print(f"ClientError: {e}"
            print(f"File {file.name} deleted")
            break
        # if file.state.name == "PROCESSING":
        sleep(1)


async def _delete_uploaded_files_async(uploaded_files, client: genai.Client):
    tasks_set = set()
    for file_name in uploaded_files.values():
        logger.info(f"Preparing to delete file: {file_name}")
        # tasks.append(asyncio.to_thread(genai.delete_file, file_name))

        # file = await client.aio.files.get(name=file_name)
        # # task = client.aio.files.delete(name=file_name)
        # task = asyncio.to_thread(client.aio.files.delete(name=file_name))
        tasks_set.add(_delete_single_uploaded_file(file_name, client))

    # await tqdm.asyncio.tqdm.gather(*tasks)
    await asyncio.gather(*tasks_set, return_exceptions=True)
    logger.info("All uploaded files deleted.")

    # async with asyncio.TaskGroup() as tg:
    #     for file_name in uploaded_files.values():
    #         logger.info(f"Preparing to delete file: {file_name}")
    #         # tasks.append(asyncio.to_thread(genai.delete_file, file_name))
    #         tg.create_task(client.aio.files.delete(name=file_name))

    # logger.info("All uploaded files deleted.")


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
