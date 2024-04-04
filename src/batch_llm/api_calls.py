import asyncio
import json

from batch_llm import Experiment, Settings
from batch_llm.models import MODELS
from src.batch_llm.utils import write_log_message

file_write_lock = asyncio.Lock()


async def query_model_and_record_response(
    prompt_dict: dict,
    settings: Settings,
    experiment: Experiment,
    index: int | str | None,
    attempt: int = 1,
) -> dict | Exception:
    """
    Send request to generate response from a LLM and record the response in a jsonl file.

    Parameters
    ----------
    prompt_dict : dict
        Dictionary containing the prompt and other parameters to be
        used for text generation. Required keys are "prompt" and "model".
        Some models may have other required keys.
    settings : Settings
        Settings for the pipeline
    experiment : Experiment
        Current experiment that is being run
    index : int | None, optional
        Integer containing the index of the prompt in the experiment,
        by default None. If None, then index is set to "NA".
        Useful for tagging the prompt/response received and any errors.
    attempt : int
        Integer containing the attempt number to process the prompt.

    Returns
    -------
    dict | Exception
        Completed prompt_dict with "response" key storing the response(s)
        from the LLM.
        A dictionary is returned if the response is received successfully or
        if the maximum number of attempts is reached (i.e. an Exception
        was caught but we have attempt==max_attempts).
        An Exception is returned if an error is caught and we have
        attempt < max_attempts, indicating that we could try this
        prompt again later in the queue.
    """
    if attempt > settings.max_attempts:
        raise ValueError(
            f"Number of attempts ({attempt}) cannot be greater than max_attempts ({settings.max_attempts})"
        )
    if index is None:
        index = "NA"

    # query the API
    timeout_seconds = 300
    # attempt to query the API max_attempts times (for timeout errors)
    # if response or another error is received, only try once and break out of the loop
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await generate_text(
                prompt_dict=prompt_dict,
                media_folder=settings.media_folder,
                log_file=experiment.log_file,
                index=index,
            )
    except Exception as err:
        if attempt == settings.max_attempts:
            # we've already tried max_attempts times, so log the error and save an error response
            log_message = (
                f"Error (i={index}) [id={prompt_dict.get('id', 'NA')}] after maximum {settings.max_attempts} attempts: "
                f"{type(err).__name__} - {err}"
            )
            write_log_message(
                log_file=experiment.log_file, log_message=log_message, log=True
            )
            response = (
                f"An unexpected error occurred when querying the API: {type(err).__name__} - {err} "
                f"after maximum {settings.max_attempts} attempts"
            )
        else:
            # we haven't tried max_attempts times yet, so log the error and return an Exception
            log_message = (
                f"Error (i={index}) [id={prompt_dict.get('id', 'NA')}] on attempt {attempt} of {settings.max_attempts}: "
                f"{type(err).__name__} - {err} - adding to the queue to try again later"
            )
            write_log_message(
                log_file=experiment.log_file, log_message=log_message, log=True
            )
            return Exception(f"{type(err).__name__} - {err}\n")

    # record the response in a jsonl file asynchronously using file_write_lock
    async with file_write_lock:
        with open(experiment.output_completed_file_path, "a") as f:
            json.dump(response, f)
            f.write("\n")

    return response


async def generate_text(
    prompt_dict: dict,
    settings: Settings,
    experiment: Experiment,
    index: int | None,
) -> dict:
    """
    Generate text by querying an LLM.

    Parameters
    ----------
    prompt_dict : dict
        Dictionary containing the prompt and other parameters to be
        used for text generation. Required keys are "prompt" and "model".
        Some models may have other required keys.
    settings : Settings
        Settings for the pipeline
    experiment : Experiment
        Current experiment that is being run
    index : int | None, optional
        Integer containing the index of the prompt in the experiment,
        by default None. If None, then index is set to "NA".
        Useful for tagging the prompt/response received and any errors.

    Returns
    -------
    dict
        Completed prompt_dict with "response" key storing the response(s)
        from the LLM.
    """
    if index is None:
        index = "NA"

    # obtain model
    model = MODELS[prompt_dict["model"]](settings=settings, experiment=experiment)

    # query the model
    response = model.async_query(prompt_dict=prompt_dict)

    return response
