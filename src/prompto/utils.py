import asyncio
import logging
import os
import shutil
from datetime import datetime

FILE_WRITE_LOCK = asyncio.Lock()


def sort_jsonl_files_by_creation_time(input_folder: str) -> list[str]:
    """
    Function sorts the jsonl files in the input folder by creation/change
    time in a given directory.

    Parameters
    ----------
    input_folder : str
        Folder which contains the files to be processed.

    Returns
    -------
    list[str]
        Ordered list of jsonl filenames in the input folder.
    """
    if not os.path.isdir(input_folder):
        raise ValueError(
            f"Input folder '{input_folder}' must be a valid path to a folder"
        )

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
        logging.info(f"Creating folder '{folder}'")
        os.makedirs(folder)
    else:
        logging.info(f"Folder '{folder}' already exists")


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
    if not os.path.exists(source):
        raise FileNotFoundError(f"File '{source}' does not exist")

    logging.info(f"Moving file from {source} to {destination}")
    os.rename(source, destination)


def copy_file(source: str, destination: str) -> None:
    """
    Function to copy a file from one location to another.

    Parameters
    ----------
    source : str
        File path of the file to be moved.
    destination : str
        File path of the destination of the file.
    """
    if not os.path.exists(source):
        raise FileNotFoundError(f"File '{source}' does not exist")

    logging.info(f"Copying file from {source} to {destination}")
    shutil.copyfile(source, destination)


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


def log_success_response_query(
    index: int | str, model: str, prompt: str, response_text: str
) -> str:
    """
    Log a successful response from a model to a query.

    Parameters
    ----------
    index : int | str
        Identifier for the query from the input file.
    model : str
        Name of the model that generated the response.
    prompt : str
        Prompt that was used to generate the response.
    response_text : str
        Response text generated by the model.

    Returns
    -------
    str
        The log message that was written.
    """
    log_message = (
        f"Response received for model {model} (i={index})\n"
        f"Prompt: {prompt[:50]}...\n"
        f"Response: {response_text[:50]}...\n"
    )
    logging.info(log_message)
    return log_message


def log_success_response_chat(
    index: int | str,
    model: str,
    message_index: int,
    n_messages: int,
    message: str,
    response_text: str,
) -> str:
    """
    Log a successful chat interaction with a model.

    Parameters
    ----------
    index : int | str
        Identifier for the query/chat from the input file.
    model : str
        Name of the model that generated the response.
    message_index : int
        Index of the message in the chat interaction.
    n_messages : int
        Total number of messages in the chat interaction.
    message : str
        Message that was sent to the model.
    response_text : str
        Response text generated by the model.

    Returns
    -------
    str
        The log message that was written.
    """
    log_message = (
        f"Response received for model {model} (i={index}, message={message_index+1}/{n_messages})\n"
        f"Prompt: {message[:50]}...\n"
        f"Response: {response_text[:50]}...\n"
    )
    logging.info(log_message)
    return log_message


def log_error_response_query(
    index: int | str, model: str, prompt: str, error_as_string: str
) -> str:
    """
    Log an error response from a model to a query.

    Parameters
    ----------
    index : int | str
        Identifier for the query from the input file.
    model : str
        Name of the model that generated the response.
    prompt : str
        Prompt that was used to generate the response.
    error_as_string : str
        Error message that was generated by the model as a string.

    Returns
    -------
    str
        The log message that was written.
    """
    log_message = (
        f"Error with model {model} (i={index})\n"
        f"Prompt: {prompt[:50]}...\n"
        f"Error: {error_as_string}"
    )
    logging.info(log_message)
    return log_message


def log_error_response_chat(
    index: int | str,
    model: str,
    message_index: int,
    n_messages: int,
    message: str,
    responses_so_far: list[str],
    error_as_string: str,
) -> str:
    """
    Log an error response from a model in a chat interaction.

    Parameters
    ----------
    index : int | str
        Identifier for the query/chat from the input file.
    model : str
        Name of the model that generated the response.
    message_index : int
        Index of the message in the chat interaction.
    n_messages : int
        Total number of messages in the chat interaction.
    message : str
        Message that was sent to the model.
    responses_so_far : list[str]
        List of responses that have been generated so far in the chat interaction.
    error_as_string : str
        Error message that was generated by the model as a string.

    Returns
    -------
    str
        The log message that was written.
    """
    log_message = (
        f"Error with model {model} (i={index}, message={message_index+1}/{n_messages})\n"
        f"Prompt: {message[:50]}...\n"
        f"Responses so far: {responses_so_far}...\n"
        f"Error: {error_as_string}"
    )
    logging.info(log_message)
    return log_message


def check_required_env_variables_set(
    required_env_variables: list[str],
) -> list[Exception]:
    """
    Check if required environment variables are set.

    A list of ValueErrors are returned for each required environment variables
    that is not set. If they are all set, an empty list is returned.

    Parameters
    ----------
    required_env_variables : list[str]
        List of environment variables that are required to be set.

    Returns
    -------
    list[Exception]
        List of exceptions that are raised if the required environment variables are not set.
    """
    return [
        ValueError(f"Environment variable '{env_variable}' is not set")
        for env_variable in required_env_variables
        if env_variable not in os.environ
    ]


def check_optional_env_variables_set(
    optional_env_variables: list[str],
) -> list[Exception]:
    """
    Check if optional environment variables are set.

    A list of Warnings are returned for each optional environment variables
    that is not set. If they are all set, an empty list is returned.

    Parameters
    ----------
    optional_env_variables : list[str]
        List of environment variables that are optional to be set.

    Returns
    -------
    list[Exception]
        List of exceptions for the optional environment variables that are not set.
    """
    return [
        Warning(f"Environment variable '{env_variable}' is not set")
        for env_variable in optional_env_variables
        if env_variable not in os.environ
    ]


def check_either_required_env_variables_set(
    required_env_variables: list[list[str]],
) -> list[Exception]:
    """
    Check if at least one of the required environment variables is set in a list
    for a given list of lists of environment variables.

    For example, if required_env_variables is `[['A', 'B'], ['C', 'D']]`,
    then we first look at `['A', 'B']`, and check at least one of the
    environment variables 'A' or 'B' are set. If either 'A' or 'B' are not set,
    we add a Warning to the returned list. IF neither 'A' or 'B' are set, we add
    a ValueError to the returned list. We then repeat this process for `['C', 'D']`.

    Parameters
    ----------
    required_env_variables : list[list[str]]
        List of lists of environment variables where at least one of the
        environment variables must be set.

    Returns
    -------
    list[Exception]
        List of exceptions of either Warnings to say an environment variable isn't set
        or ValueErrors if none of the required environment variables in a list are set.
    """
    # check required environment variables is a list of lists
    if not all(
        isinstance(env_variables, list) for env_variables in required_env_variables
    ):
        raise TypeError(
            "The 'required_env_variables' parameter must be a list of lists of environment variables"
        )

    issues = []
    for env_variables in required_env_variables:
        # see what variables are not set and get a list of Warnings
        warnings = check_optional_env_variables_set(env_variables)

        if len(warnings) == len(env_variables):
            # add a value error if none of the variables in this list are set
            issues.append(
                ValueError(
                    f"At least one of the environment variables '{env_variables}' must be set"
                )
            )
        else:
            # add the warnings to the list of issues if at least one variable is set
            issues.extend(warnings)

    return issues


def get_model_name_identifier(model_name: str) -> str:
    """
    Helper function to get the model name identifier.

    Some model names can contain characters that are not allowed in
    environment variable names. This function replaces those characters
    ("-", "/", ".", ":", " ") with underscores ("_").

    Parameters
    ----------
    model_name : str
        The model name

    Returns
    -------
    str
        The model name identifier with invalid characters replaced
        with underscores
    """
    model_name = model_name.replace("-", "_")
    model_name = model_name.replace("/", "_")
    model_name = model_name.replace(".", "_")
    model_name = model_name.replace(":", "_")
    model_name = model_name.replace(" ", "_")

    return model_name


def sort_prompts_by_model_for_api(prompt_dicts: list[dict], api: str) -> list[dict]:
    """
    For a list of prompt dictionaries, sort the dictionaries with `"api": api`
    by the "model_name" key. The rest of the dictionaries are kept in the same order.

    For Ollama API, if the model requested is not currently loaded, the model will be
    loaded on demand. This can take some time, so it is better to sort the prompts
    by the model name to reduce the time taken to load the models.

    If no dictionaries with `"api": api` are present, the original list is returned.

    Parameters
    ----------
    prompt_dicts : list[dict]
        List of dictionaries containing the prompt and other parameters
        to be sent to the API. Each dictionary must have keys "prompt" and "api"
    api : str
        The API name to sort the prompt dictionaries by the "model_name" key

    Returns
    -------
    list[dict]
        List of dictionaries containing the prompt and other parameters
        where the dictionaries with `"api": api` are sorted by the "model_name" key
    """
    api_indices = [i for i, item in enumerate(prompt_dicts) if item.get("api") == api]
    if len(api_indices) == 0:
        return prompt_dicts

    # sort indices for dictionaries with "api": api
    sorted_api_indices = sorted(
        api_indices, key=lambda i: prompt_dicts[i].get("model_name", "")
    )

    # create map from original api index to sorted index
    api_index_map = {i: j for i, j in zip(api_indices, sorted_api_indices)}

    # sort data based on the combined indices
    return [
        (
            prompt_dicts[i]
            if i not in api_index_map.keys()
            else prompt_dicts[api_index_map[i]]
        )
        for i in range(len(prompt_dicts))
    ]
