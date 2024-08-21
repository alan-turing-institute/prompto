import asyncio
import json
import logging
import os
import time
from datetime import datetime

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from prompto.apis import ASYNC_APIS
from prompto.settings import Settings
from prompto.utils import (
    FILE_WRITE_LOCK,
    create_folder,
    move_file,
    sort_prompts_by_model_for_api,
    write_log_message,
)

TIMESTAMP_FORMAT = "%d-%m-%Y-%H-%M-%S"


class Experiment:
    """
    A class to represent an experiment. An experiment is a jsonl file
    containing a list of prompts to be sent to a language model.

    An Experiment is also ran with a set of settings for the pipeline
    to run the experiment.

    Parameters
    ----------
    file_name : str
        The name of the jsonl experiment file
    settings : Settings
        Settings for the pipeline which includes the data folder locations,
        the maximum number of queries to send per minute, the maximum number
        of attempts when retrying, and whether to run the experiment in parallel
    """

    def __init__(
        self,
        file_name: str,
        settings: Settings,
    ):
        if not file_name.endswith(".jsonl"):
            raise ValueError("Experiment file must be a jsonl file")

        self.file_name: str = file_name
        # obtain experiment name from file name
        self.experiment_name: str = self.file_name.removesuffix(".jsonl")
        # settings for the pipeline which includes input, output, and media folder locations
        self.settings: Settings = settings
        # experiment output folder is a subfolder of the output folder
        self.output_folder: str = os.path.join(
            self.settings.output_folder, self.experiment_name
        )

        # obtain file paths
        # file path to the original input file
        self.input_file_path: str = os.path.join(
            self.settings.input_folder, self.file_name
        )

        # check that the experiment file exists
        if not os.path.exists(self.input_file_path):
            raise FileNotFoundError(
                f"Experiment file '{self.input_file_path}' does not exist"
            )

        # read in the experiment data
        with open(self.input_file_path, "r") as f:
            self._experiment_prompts: list[dict] = [
                dict(json.loads(line)) for line in f
            ]
            # sort the prompts by model_name key for the ollama api
            # (for avoiding constantly switching and loading models between prompts)
            self._experiment_prompts = sort_prompts_by_model_for_api(
                self._experiment_prompts, api="ollama"
            )

        # set the number of queries
        self.number_queries: int = len(self._experiment_prompts)

        # get the time which the experiment file is created
        self.creation_time: str = datetime.fromtimestamp(
            os.path.getctime(self.input_file_path)
        ).strftime(TIMESTAMP_FORMAT)

        # get the current time of when the experiment has started to run including time zone
        self.start_time = datetime.now().strftime(TIMESTAMP_FORMAT)

        # log file is a file in the experiment output folder
        self.log_file: str = os.path.join(
            self.output_folder, f"{self.start_time}-log-{self.experiment_name}.txt"
        )
        # file path of the completed experiment file in the output experiment folder
        self.output_completed_file_path: str = os.path.join(
            self.output_folder, f"{self.start_time}-completed-" + self.file_name
        )
        # file path of the input file in the output experiment folder (for logging purposes)
        self.output_input_file_out_path: str = os.path.join(
            self.output_folder, f"{self.start_time}-input-" + self.file_name
        )

        # grouped experiment prompts by
        # only group the prompts on the first call to the property
        self._grouped_experiment_prompts: dict[str, list[dict]] = {}

        # initialise the completed responses
        self.completed_responses: list[dict] = []

    def __str__(self) -> str:
        return self.file_name

    @property
    def experiment_prompts(self) -> list[dict]:
        return self._experiment_prompts

    @experiment_prompts.setter
    def experiment_prompts(self, value: list[dict]) -> None:
        raise AttributeError("Cannot set the experiment_prompts attribute")

    @property
    def grouped_experiment_prompts(self) -> dict[str, list[dict]]:
        # if settings.parallel is False, then we won't utilise the grouping
        if not self.settings.parallel:
            logging.warning(
                "The 'parallel' attribute in the Settings object is set to False, "
                "so grouping will not be used when processing the experiment prompts. "
                "Set 'parallel' to True to use grouping and parallel processing of prompts."
            )

        # only group the prompts on the first call to the property
        # i.e. we only group the experiment prompts when we need to
        if self._grouped_experiment_prompts == {}:
            self._grouped_experiment_prompts = self.group_prompts()

        return self._grouped_experiment_prompts

    @grouped_experiment_prompts.setter
    def grouped_experiment_prompts(self, value: dict[str, list[dict]]) -> None:
        raise AttributeError("Cannot set the grouped_experiment_prompts attribute")

    def group_prompts(self) -> dict[str, list[dict]]:
        """
        Function to group the experiment prompts by either the "group" key
        or the "api" key in the prompt dictionaries. The "group" key is
        used if it exists, otherwise the "api" key is used.

        Depending on the 'max_queries_dict' attribute in the settings object
        (of class Settings), the prompts may also be further split by
        the model name (if a model-specific rate limit is provided).

        It first initialises a dictionary with keys as the grouping names
        determined by the 'max_queries_dict' attribute in the settings object,
        and values are dictionaries with "prompt_dicts" and "rate_limit" keys.
        It will use any of the rate limits provided to intialise these values.
        The function then loops over the experiment prompts and adds them to the
        appropriate group in the dictionary. If a grouping name (given by the "group" or
        "api" key) is not in the dictionary already, it will initialise it
        with an empty list of prompt dictionaries and the default rate limit
        (given by the 'max_queries' attribute in the settings).

        Returns
        -------
        dict[str, dict[str, list[dict] | int]
            Dictionary where the keys are the grouping names (either a group name
            or an API name, and potentially with a model name tag too) and the values
            are dictionaries with "prompt_dicts" and "rate_limit" keys. The "prompt_dicts"
            key stores a list of prompt dictionaries for that group, and the "rate_limit"
            key stores the maximum number of queries to send per minute for that group
        """
        grouped_dict = {}
        # initialise some keys with the rate limits if provided
        if self.settings.max_queries_dict != {}:
            logging.info(
                "Grouping prompts using 'settings.max_queries_dict': "
                f"{self.settings.max_queries_dict}..."
            )
            for key, value in self.settings.max_queries_dict.items():
                if isinstance(value, int):
                    # a default was provided for this api / group
                    grouped_dict[key] = {
                        "prompt_dicts": [],
                        "rate_limit": value,
                    }
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        # sub_key is the model name (or "default")
                        # sub_value is the rate limit for that model
                        # (or the default for the api / group)
                        if sub_key == "default":
                            # a default was provided for this api / group
                            grouped_dict[key] = {
                                "prompt_dicts": [],
                                "rate_limit": sub_value,
                            }
                        else:
                            # a model-specific rate for the api / group was provided
                            grouped_dict[f"{key}-{sub_key}"] = {
                                "prompt_dicts": [],
                                "rate_limit": sub_value,
                            }
        else:
            logging.info("Grouping prompts by 'group' or 'api' key...")

        # add the prompts to the grouped dictionary
        for prompt_dict in self._experiment_prompts:
            # obtain the key to add the prompt_dict to
            # "group" key is used if it exists, otherwise use "api"
            if "group" in prompt_dict:
                key = prompt_dict["group"]
            else:
                key = prompt_dict["api"]

            if key not in grouped_dict:
                # initilise the key with an empty prompt_dicts list
                # and the rate limit is just the default max_queries
                # as no rate limit was provided for this api / group
                grouped_dict[key] = {
                    "prompt_dicts": [],
                    "rate_limit": self.settings.max_queries,
                }

            # model-specific rates may have been provided in the settings
            if key in self.settings.max_queries_dict and isinstance(
                self.settings.max_queries_dict[key], dict
            ):
                if prompt_dict.get("model_name") in self.settings.max_queries_dict[key]:
                    key = f"{key}-{prompt_dict.get('model_name')}"

                if key not in grouped_dict:
                    # initialise model-specific key
                    grouped_dict[key] = {
                        "prompt_dicts": [],
                        "rate_limit": self.settings.max_queries,
                    }

            grouped_dict[key]["prompt_dicts"].append(prompt_dict)

        return grouped_dict

    def grouped_experiment_prompts_summary(self) -> dict[str, str]:
        """
        Generate a dictionary with the group names as keys
        and the number of queries and rate limit for each group
        as a string.

        Returns
        -------
        dict[str, str]
            Dictionary with the group names as keys and the number
            of queries and rate limit for each group as a string
        """
        queries_and_rates_per_group = {
            group: f"{len(values['prompt_dicts'])} queries at {values['rate_limit']} queries per minute"
            for group, values in self.grouped_experiment_prompts.items()
        }

        return queries_and_rates_per_group

    async def process(self) -> tuple[dict, float]:
        """
        Function to process the experiment.

        The method will first create a folder for the experiment in the output
        folder named after the experiment name (filename without the .jsonl extension).
        It will then move the input experiment file to the output folder.

        The method will then send the prompts to the API asynchronously and
        record the responses in an output jsonl file in the output experiment folder.
        Logs will be printed and saved in the log file for the experiment.

        All output files are timestamped with the time for when the experiment
        started to run.

        Returns
        -------
        tuple[dict, float]
            A tuple containing the completed prompt_dicts from the API and the
            average processing time per query for the experiment
        """
        logging.info(f"Processing experiment: {self.__str__()}...")
        start_time = time.time()

        # create the output folder for the experiment
        create_folder(self.output_folder)

        # move the experiment file to the output folder
        logging.info(
            f"Moving {self.input_file_path} to {self.output_folder} as "
            f"{self.output_input_file_out_path}..."
        )
        move_file(
            source=self.input_file_path,
            destination=self.output_input_file_out_path,
        )

        # run the experiment asynchronously
        if self.settings.parallel:
            logging.info(
                f"Sending {self.number_queries} queries in parallel by grouping prompts..."
            )
            logging.info(
                f"Queries per group: {self.grouped_experiment_prompts_summary()}"
            )

            # create tasks for each group which we will run in parallel using asyncio.gather
            tasks = [
                asyncio.create_task(
                    self.send_requests_retry(
                        prompt_dicts=values["prompt_dicts"],
                        group=group,
                        rate_limit=values["rate_limit"],
                    )
                )
                for group, values in self.grouped_experiment_prompts.items()
            ]
            await tqdm_asyncio.gather(
                *tasks, desc="Waiting for all groups to complete", unit="group"
            )
        else:
            logging.info(f"Sending {self.number_queries} queries...")
            await self.send_requests_retry(
                prompt_dicts=self.experiment_prompts,
                group=None,
                rate_limit=self.settings.max_queries,
            )

        # calculate average processing time per query for the experiment
        end_time = time.time()
        processing_time = end_time - start_time
        avg_query_processing_time = processing_time / self.number_queries

        # log completion of experiment
        log_message = (
            f"Completed experiment {self.__str__()}! "
            f"Experiment processing time: {round(processing_time, 3)} seconds, "
            f"Average time per query: {round(avg_query_processing_time, 3)} seconds"
        )
        async with FILE_WRITE_LOCK:
            write_log_message(log_file=self.log_file, log_message=log_message, log=True)

        # read the output file
        with open(self.output_completed_file_path, "r") as f:
            self.completed_responses: list[dict] = [
                dict(json.loads(line)) for line in f
            ]

        return self.completed_responses, avg_query_processing_time

    async def send_requests(
        self,
        prompt_dicts: list[dict],
        attempt: int,
        rate_limit: int,
        group: str | None = None,
    ) -> tuple[list[dict], list[dict | Exception]]:
        """
        Send requests to the API asynchronously.

        The method will send the prompts to the API asynchronously with a wait
        interval between requests in order to not exceed the maximum number of
        queries per minute specified by the experiment settings.

        For each prompt_dict in prompt_dicts, the method will query the model
        and record the response in a jsonl file if successful. If the query fails,
        an Exception is returned.

        A tuple is returned containing the input prompt_dicts and their corresponding
        completed prompt_dicts with the responses from the API. For any failed queries,
        the response will be an Exception.

        This tuple can be used to determine easily which prompts failed and potentially
        need to be retried.

        Parameters
        ----------
        prompt_dicts : list[dict]
            List of dictionaries containing the prompt and other parameters
            to be sent to the API. Each dictionary must have keys "prompt" and "api".
            Optionally, they can have a "parameters" key. Some APIs may have
            other specific required keys
        attempt : int
            The attempt number to process the prompt
        rate_limit : int
            The maximum number of queries to send per minute
        group : str | None, optional
            Group name, by default None. If None, then the group is
            not specified in the logs

        Returns
        -------
        tuple[list[dict], list[dict | Exception]]
            A tuple containing the input prompt_dicts and their corresponding
            responses (given in the form of completed prompt_dicts, i.e. a
            prompt_dict with a completed "response" key) from the API.
            For any failed queries, the response will be an Exception.
        """
        request_interval = 60 / rate_limit
        tasks = []
        for_group_string = f"for group '{group}' " if group is not None else ""
        attempt_frac = f"{attempt}/{self.settings.max_attempts}"

        for index, item in enumerate(
            tqdm(
                prompt_dicts,
                desc=(
                    f"Sending {len(prompt_dicts)} queries at {rate_limit} QPM with RI of "
                    f"{request_interval}s {for_group_string}(attempt {attempt_frac})"
                ),
                unit="query",
            )
        ):
            # wait interval between requests
            await asyncio.sleep(request_interval)

            # query the API asynchronously and collect the task
            task = asyncio.create_task(
                self.query_model_and_record_response(
                    prompt_dict=item,
                    index=index + 1,
                    attempt=attempt,
                )
            )
            tasks.append(task)

        # wait for all tasks to complete before returning
        responses = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Waiting for responses {for_group_string}(attempt {attempt_frac})",
            unit="query",
        )

        return prompt_dicts, responses

    async def send_requests_retry(
        self,
        prompt_dicts: list[dict],
        rate_limit: int,
        group: str | None = None,
    ) -> None:
        """
        Send requests to the API asynchronously and retry failed queries
        up to a maximum number of attempts.

        Wrapper function around send_requests that retries failed queries
        for a maximum number of attempts specified by the experiment settings
        or until all queries are successful.

        Parameters
        ----------
        prompt_dicts : list[dict]
            List of dictionaries containing the prompt and other parameters
            to be sent to the API. Each dictionary must have keys "prompt" and "api".
            Optionally, they can have a "parameters" key. Some APIs may have
            other specific required keys
        group : str | None, optional
            Group name, by default None. If None, then the group is
            not specified in the logs
        """
        # initialise the number of attempts
        attempt = 1

        # send off the requests
        remaining_prompt_dicts, responses = await self.send_requests(
            prompt_dicts=prompt_dicts,
            attempt=attempt,
            rate_limit=rate_limit,
            group=group,
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
                        f"Retrying {len(remaining_prompt_dicts)} failed queries - "
                        f"attempt {attempt} of {self.settings.max_attempts}..."
                    )

                    # send off the failed queries
                    remaining_prompt_dicts, responses = await self.send_requests(
                        prompt_dicts=remaining_prompt_dicts,
                        attempt=attempt,
                        rate_limit=rate_limit,
                        group=group,
                    )
                else:
                    # if there are no failed queries, break out of the loop
                    break
            else:
                # if the maximum number of attempts has been reached, break out of the loop
                break

    async def query_model_and_record_response(
        self,
        prompt_dict: dict,
        index: int | str | None,
        attempt: int,
    ) -> dict | Exception:
        """
        Send request to generate response from a LLM and record the response in a jsonl file.

        Parameters
        ----------
        prompt_dict : dict
            Dictionary containing the prompt and other parameters to be
            used for text generation. Required keys are "prompt" and "api".
            Optionally can have a "parameters" key. Some APIs may have
            other specific required keys
        index : int | None, optional
            The index of the prompt in the experiment,
            by default None. If None, then index is set to "NA".
            Useful for tagging the prompt/response received and any errors
        attempt : int
            The attempt number to process the prompt

        Returns
        -------
        dict | Exception
            Completed prompt_dict with "response" key storing the response(s)
            from the LLM.
            A dictionary is returned if the response is received successfully or
            if the maximum number of attempts is reached (i.e. an Exception
            was caught but we have attempt==max_attempts).
            An Exception is returned (not raised) if an error is caught and we have
            attempt < max_attempts, indicating that we could try this
            prompt again later in the queue.
        """
        if attempt > self.settings.max_attempts:
            raise ValueError(
                f"Attempt number ({attempt}) cannot be greater than "
                f"settings.max_attempts ({self.settings.max_attempts})"
            )
        if index is None:
            index = "NA"

        # query the API
        timeout_seconds = 300
        # attempt to query the API max_attempts times (for timeout errors)
        # if response or another error is received, only try once and break out of the loop
        try:
            async with asyncio.timeout(timeout_seconds):
                completed_prompt_dict = await self.generate_text(
                    prompt_dict=prompt_dict,
                    index=index,
                )
        except (NotImplementedError, KeyError, ValueError, TypeError) as err:
            # don't retry for selected errors, log the error and save an error response
            log_message = (
                f"Error (i={index}, id={prompt_dict.get('id', 'NA')}): "
                f"{type(err).__name__} - {err}"
            )
            async with FILE_WRITE_LOCK:
                write_log_message(
                    log_file=self.log_file, log_message=log_message, log=True
                )
            # fill in response with error message
            completed_prompt_dict = prompt_dict
            completed_prompt_dict["response"] = f"{type(err).__name__} - {err}"
        except (Exception, asyncio.CancelledError, asyncio.TimeoutError) as err:
            if attempt == self.settings.max_attempts:
                # we've already tried max_attempts times, so log the error and save an error response
                log_message = (
                    f"Error (i={index}, id={prompt_dict.get('id', 'NA')}) "
                    f"after maximum {self.settings.max_attempts} attempts: "
                    f"{type(err).__name__} - {err}"
                )
                async with FILE_WRITE_LOCK:
                    write_log_message(
                        log_file=self.log_file, log_message=log_message, log=True
                    )
                # fill in response with error message and note that we've tried max_attempts times
                completed_prompt_dict = prompt_dict
                completed_prompt_dict["response"] = (
                    "An unexpected error occurred when querying the API: "
                    f"({type(err).__name__} - {err}) "
                    f"after maximum {self.settings.max_attempts} attempts"
                )
            else:
                # we haven't tried max_attempts times yet, so log the error and return an Exception
                log_message = (
                    f"Error (i={index}, id={prompt_dict.get('id', 'NA')}) on attempt "
                    f"{attempt} of {self.settings.max_attempts}: "
                    f"{type(err).__name__} - {err}. Adding to the queue to try again later..."
                )
                async with FILE_WRITE_LOCK:
                    write_log_message(
                        log_file=self.log_file, log_message=log_message, log=True
                    )
                # return Execption to indicate that we should try this prompt again later
                return Exception(f"{type(err).__name__} - {err}\n")

        # record the response in a jsonl file asynchronously using FILE_WRITE_LOCK
        async with FILE_WRITE_LOCK:
            with open(self.output_completed_file_path, "a") as f:
                json.dump(completed_prompt_dict, f)
                f.write("\n")

        return completed_prompt_dict

    async def generate_text(
        self,
        prompt_dict: dict,
        index: int | None,
    ) -> dict:
        """
        Generate text by querying an LLM.

        Parameters
        ----------
        prompt_dict : dict
            Dictionary containing the prompt and other parameters to be
            used for text generation. Required keys are "prompt" and "api".
            Some models may have other required keys.
        index : int | None, optional
            The index of the prompt in the experiment,
            by default None. If None, then index is set to "NA".
            Useful for tagging the prompt/response received and any errors

        Returns
        -------
        dict
            Completed prompt_dict with "response" key storing the response(s)
            from the LLM
        """
        if index is None:
            index = "NA"
        if "api" not in prompt_dict:
            raise KeyError(
                "API is not specified in the prompt_dict. Must have 'api' key"
            )

        # obtain api class
        try:
            api = ASYNC_APIS[prompt_dict["api"]](
                settings=self.settings, log_file=self.log_file
            )
        except KeyError:
            raise NotImplementedError(
                f"API {prompt_dict['api']} not recognised or implemented"
            )

        # add a timestamp to the prompt_dict
        prompt_dict["timestamp_sent"] = datetime.now().strftime(TIMESTAMP_FORMAT)

        # query the model
        response = await api.query(prompt_dict=prompt_dict, index=index)

        return response
