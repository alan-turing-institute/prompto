import logging
import os
from datetime import datetime


def sort_jsonl_files_by_creation_time(input_folder: str) -> list[str]:
    """
    Function sorts the jsonl files in the input folder by creation time
    in a given directory.

    Parameters
    ----------
    input_folder : str
        Folder which contains the files to be processed.

    Returns
    -------
    list[str]
        Ordered list of jsonl filenames in the input folder.
    """
    return sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".jsonl")],
        key=lambda f: os.path.getctime(os.path.join(input_folder, f)),
    )


def create_folder(folder: str) -> None:
    """
    Function to create a folder if it does not already exist.

    Parameters
    ----------
    folder : str
        Name of the folder to be created.
    """
    if not os.path.exists(folder):
        logging.info(f"Creating folder {folder}")
        os.makedirs(folder)
    else:
        logging.info(f"Folder {folder} already exists")


def move_file(source: str, destination: str) -> None:
    """
    Function to move a file from one location to another.

    Parameters
    ----------
    source : str
        File path of the file to be moved.
    destination : str
        File path of the destination of the file.
    """
    logging.info(f"Moving file from {source} to {destination}")
    os.rename(source, destination)


def write_log_message(log_file: str, log_message: str, log: bool = True) -> None:
    """
    Helper function to write a log message to a log file
    with the current date and time of the log message.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    log_message : str
        Message to be written to the log file.
    """
    if log:
        logging.info(log_message)

    now = datetime.now()
    with open(log_file, "a") as log:
        log.write(f"{now.strftime('%d-%m-%Y, %H:%M')}: {log_message}\n")


def log_success_response_query(index, model, prompt, response_text):
    log_message = (
        f"Response recieved for model {model} (i={index}) \nPrompt: {prompt[:50]}... \n"
        f"Response: {response_text[:50]}...\n"
    )
    logging.info(log_message)
    return log_message


def log_success_response_chat(
    index, model, message_index, n_messages, message, response_text
):
    log_message = (
        f"Response recieved for model {model} (i={index}, message={message_index+1}/{n_messages}) "
        f"\nPrompt: {message[:50]}... \nResponse: {response_text[:50]}...\n"
    )
    logging.info(log_message)
    return log_message


def log_error_response_query(index, model, prompt, error_as_string):
    log_message = (
        f"Error with {model} model (i={index}): "
        f"\nPrompt: {prompt[:50]}... \nError: {error_as_string}"
    )
    logging.info(log_message)
    return log_message


def log_error_response_chat(
    index, model, message_index, message, responses_so_far, error_as_string
):
    log_message = (
        f"Error with {model} chat model (i={index}, at message {message_index+1}): "
        f"\nPrompt: {message[:50]}... "
        f"\nResponses so far: {responses_so_far}... \nError: {error_as_string}"
    )
    logging.info(log_message)
    return log_message
