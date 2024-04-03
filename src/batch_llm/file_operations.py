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
    os.rename(source, destination)


def write_log_message(log_file: str, log_message: str) -> None:
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
    now = datetime.now()
    with open(log_file, "a") as log:
        log.write(f"{now.strftime('%d-%m-%Y, %H:%M')}: {log_message}\n")
