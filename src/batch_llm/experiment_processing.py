import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from batch_llm.models import ASYNC_MODELS
from batch_llm.settings import Settings
from batch_llm.utils import (
    create_folder,
    move_file,
    sort_jsonl_files_by_creation_time,
    write_log_message,
)

file_write_lock = asyncio.Lock()


class Experiment:
    def __init__(
        self,
        file_name: str,
        settings: Settings,
    ):
        if not file_name.endswith(".jsonl"):
            raise ValueError("Experiment file must be a jsonl file.")

        self.file_name: str = file_name
        # obtain experiment name from file name
        self.experiment_name: str = self.file_name.removesuffix(".jsonl")
        # settings for the pipeline which includes input, output, and media folder locations
        self.settings: Settings = settings
        # experiment output folder is a subfolder of the output folder
        self.output_folder: str = os.path.join(
            self.settings.output_folder, self.experiment_name
        )
        # get the time which the experiment file is created
        self.creation_time: str = datetime.fromtimestamp(
            os.path.getctime(os.path.join(self.settings.input_folder, self.file_name))
        ).strftime("%d-%m-%Y-%H-%M")
        # log file is a file in the experiment output folder
        self.log_file: str = os.path.join(
            self.output_folder, f"{self.creation_time}-log.txt"
        )

        # obtain file paths
        # file path to the original input file
        self.input_file_path: str = os.path.join(
            self.settings.input_folder, self.file_name
        )
        # file path of the completed experiment file in the output experiment folder
        self.output_completed_file_path: str = os.path.join(
            self.output_folder, "completed-" + self.file_name
        )
        # file path of the input file in the output experiment folder (for logging purposes)
        self.output_input_file_out_path: str = os.path.join(
            self.output_folder, "input-" + self.file_name
        )

        # read in the experiment data
        with open(self.input_file_path, "r") as f:
            self.experiment_prompts: list[dict] = [dict(json.loads(line)) for line in f]
        # set the number of queries
        self.number_queries: int = len(self.experiment_prompts)

    def __str__(self) -> str:
        return self.file_name


class ExperimentPipeline:
    def __init__(
        self,
        settings: Settings,
    ):
        self.settings: Settings = settings
        self.average_per_query_processing_times: list[float] = []
        self.overall_avg_proc_times: float = 0.0
        self.experiment_files: list[str] = []

    def run(self) -> None:
        """
        Run the pipeline process of continually checking for new experiment files
        and running the experiments sequentially.
        """
        while True:
            # obtain experiment files sorted by creation time
            self.update_experiment_files()

            if len(self.experiment_files) != 0:
                # obtain the next experiment to process
                next_experiment = Experiment(
                    file_name=self.experiment_files[0], settings=self.settings
                )

                # proccess the next experiment
                asyncio.run(self.process_experiment(experiment=next_experiment))

                # log the progress of the queue of experiments
                self.log_progress(experiment=next_experiment)

    def update_experiment_files(self) -> None:
        # get the list of experiment files
        self.experiment_files = sort_jsonl_files_by_creation_time(
            input_folder=self.settings.input_folder
        )

    def log_estimate(
        self,
        experiment: Experiment,
    ) -> None:
        """
        Function to log the estimated time of completion of the next experiment.
        """
        now = datetime.now()
        if self.overall_avg_proc_times == 0:
            estimated_completion = "[unknown]"
        else:
            estimated_completion = (
                now
                + timedelta(
                    seconds=round(
                        self.overall_avg_proc_times * experiment.number_queries, 3
                    )
                )
            ).strftime("%d-%m-%Y, %H:%M")

        # log the estimated time of completion of the next experiment
        log_message = (
            f"Next experiment: {experiment}, "
            f"Number queries: {experiment.number_queries}, "
            f"Estimated completion by: {estimated_completion}"
        )
        write_log_message(log_file=experiment.log_file, log_message=log_message)

    def log_progress(
        self,
        experiment: Experiment,
    ) -> None:
        """
        Function to log the progress of the queue of experiments.
        """
        # log completion of experiment
        logging.info(f"Completed experiment: {experiment}!")
        logging.info(
            f"- Overall average time per query: {round(self.overall_avg_proc_times, 3)} seconds"
        )

        # log remaining of experiments
        self.update_experiment_files()
        logging.info(f"- Remaining number of experiments: {len(self.experiment_files)}")
        logging.info(f"- Remaining experiments: {self.experiment_files}")

    async def process_experiment(
        self,
        experiment: Experiment,
    ) -> None:
        """
        Function to process the next experiment in the queue.
        """
        logging.info(f"Processing experiment: {experiment}...")
        start_time = time.time()

        # create the output folder for the experiment
        create_folder(experiment.output_folder)

        # move the experiment file to the output folder
        logging.info(
            f"Moving {experiment.input_file_path} to {experiment.output_folder} as "
            f"{experiment.output_input_file_out_path}..."
        )
        move_file(
            source=experiment.input_file_path,
            destination=experiment.output_input_file_out_path,
        )

        # log the estimated time of completion of the next experiment
        self.log_estimate(experiment=experiment)

        # run the experiment asynchronously
        logging.info(f"Sending {experiment.number_queries} queries...")
        await self.send_requests_retry(
            experiment=experiment,
        )

        # calculate average processing time per query for the experiment
        end_time = time.time()
        processing_time = end_time - start_time
        avg_query_processing_time = processing_time / experiment.number_queries

        # log completion of experiment
        log_message = (
            f"Completed experiment {experiment}! "
            f"Experiment processing time: {round(processing_time, 3)} seconds, "
            f"Average time per query: {round(avg_query_processing_time, 3)} seconds"
        )
        write_log_message(
            log_file=experiment.log_file, log_message=log_message, log=True
        )

        # keep track of the average processing time per query for the experiment
        self.average_per_query_processing_times.append(avg_query_processing_time)

        # update the overall average processing time per query
        self.overall_avg_proc_times = sum(
            self.average_per_query_processing_times
        ) / len(self.average_per_query_processing_times)

    async def send_requests(
        self,
        experiment: Experiment,
        prompt_dicts: list[dict],
        attempt: int,
    ) -> tuple[list[dict], list[dict | Exception]]:
        """
        Send requests to the API asynchronously.
        """
        request_interval = 60 / self.settings.max_queries
        tasks = []

        for index, item in enumerate(
            tqdm(
                prompt_dicts,
                desc=f"Sending {len(prompt_dicts)} queries",
                unit="query",
            )
        ):
            # wait interval between requests
            await asyncio.sleep(request_interval)

            # query the API asynchronously and collect the task
            task = asyncio.create_task(
                query_model_and_record_response(
                    prompt_dict=item,
                    settings=self.settings,
                    experiment=experiment,
                    index=index + 1,
                    attempt=attempt,
                )
            )
            tasks.append(task)

        # wait for all tasks to complete before returning
        responses = await tqdm_asyncio.gather(
            *tasks, desc="Waiting for responses", unit="query"
        )

        return prompt_dicts, responses

    async def send_requests_retry(
        self,
        experiment: Experiment,
    ) -> None:
        """
        Send requests to the API asynchronously and retry failed queries
        up to a maximum number of attempts.
        """
        # initialise the number of attempts
        attempt = 1

        # send off the requests
        remaining_prompt_dicts, responses = await self.send_requests(
            experiment=experiment,
            prompt_dicts=experiment.experiment_prompts,
            attempt=attempt,
        )

        while True:
            # increment the attempt number
            attempt += 1
            if attempt <= self.settings.max_attempts:
                # filter the failed queries
                remaining_prompt_dicts = [
                    prompt
                    for prompt, resp in zip(remaining_prompt_dicts, responses)
                    if isinstance(resp, Exception)
                ]

                # if we still have failed queries, we will retry them
                if len(remaining_prompt_dicts) > 0:
                    logging.info(
                        f"Retrying {len(remaining_prompt_dicts)} failed queries - attempt {attempt} of "
                        f"{self.settings.max_attempts}..."
                    )

                    # send off the failed queries
                    remaining_prompt_dicts, responses = await self.send_requests(
                        experiment=experiment,
                        prompt_dicts=remaining_prompt_dicts,
                        attempt=attempt,
                    )
                else:
                    # if there are no failed queries, break out of the loop
                    break
            else:
                # if the maximum number of attempts has been reached, break out of the loop
                break


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
            completed_prompt_dict = await generate_text(
                prompt_dict=prompt_dict,
                settings=settings,
                experiment=experiment,
                index=index,
            )
    except (NotImplementedError, KeyError, ValueError, TypeError) as err:
        # don't retry for selected errors, log the error and save an error response
        log_message = (
            f"Error (i={index}) [id={prompt_dict.get('id', 'NA')}]. "
            f"{type(err).__name__} - {err}"
        )
        write_log_message(
            log_file=experiment.log_file, log_message=log_message, log=True
        )
        # fill in response with error message
        completed_prompt_dict = prompt_dict
        completed_prompt_dict["response"] = f"{type(err).__name__} - {err}"
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
            # fill in response with error message and note that we've tried max_attempts times
            completed_prompt_dict = prompt_dict
            completed_prompt_dict["response"] = (
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
            # return Execption to indicate that we should try this prompt again later
            return Exception(f"{type(err).__name__} - {err}\n")

    # record the response in a jsonl file asynchronously using file_write_lock
    async with file_write_lock:
        with open(experiment.output_completed_file_path, "a") as f:
            json.dump(completed_prompt_dict, f)
            f.write("\n")

    return completed_prompt_dict


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
    if "model" not in prompt_dict:
        raise KeyError(
            "Model is not specified in the prompt_dict. Must have 'model' key"
        )

    # obtain model
    try:
        model = ASYNC_MODELS[prompt_dict["model"]](
            settings=settings, log_file=experiment.log_file
        )
    except KeyError:
        raise NotImplementedError(
            f"Model {prompt_dict['model']} not recognised or implemented"
        )

    # query the model
    response = await model.async_query(prompt_dict=prompt_dict, index=index)

    return response
