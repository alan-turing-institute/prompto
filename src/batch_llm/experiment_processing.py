import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from batch_llm import Settings
from batch_llm.api_calls import query_model_and_record_response
from src.batch_llm.utils import (
    create_folder,
    move_file,
    sort_jsonl_files_by_creation_time,
    write_log_message,
)


class Experiment:
    def __init__(
        self,
        file_name: str,
        settings: Settings,
    ):
        if not file_name.endswith(".jsonl"):
            raise ValueError("Experiment file must be a jsonl file.")

        self.file_name = file_name
        # obtain experiment name from file name
        self.experiment_name = self.file_name.removesuffix(".jsonl")
        # settings for the pipeline which includes input, output, and media folder locations
        self.settings = settings
        # experiment output folder is a subfolder of the output folder
        self.output_folder = os.path.join(
            self.settings.output_folder, self.experiment_name
        )
        # get the time which the experiment file is created
        self.creation_time = datetime.fromtimestamp(
            os.path.getctime(os.path.join(self.settings.input_folder, self.file_name))
        ).strftime("%d-%m-%Y-%H-%M")
        # log file is a file in the experiment output folder
        self.log_file = os.path.join(
            self.output_folder, f"{self.creation_time}-log.txt"
        )

        # obtain file paths
        # file path to the original input file
        self.input_file_path = os.path.join(self.settings.input_folder, self.file_name)
        # file path of the completed experiment file in the output experiment folder
        self.output_completed_file_path = os.path.join(
            self.output_folder, "completed-" + self.file_name
        )
        # file path of the input file in the output experiment folder (for logging purposes)
        self.output_input_file_out_path = os.path.join(
            self.output_folder, "input-" + self.file_name
        )

        # read in the experiment data
        with open(self.in_file_out_path, "r") as f:
            self.experiment_prompts = [json.loads(line) for line in f]
        # set the number of queries
        self.number_queries = len(self.experiment_prompts)

    def __str__(self) -> str:
        return self.file_name

    # settings is read only
    @property
    def settings(self) -> Settings:
        return self.settings


class ExperimentPipeline:
    def __init__(
        self,
        settings: Settings,
    ):
        self.settings: Settings = settings
        self.average_per_query_processing_times: list[float] = []
        self.overall_avg_proc_times: float = 0.0
        self.experiment_files: list[str] = []

    # settings is read only
    @property
    def settings(self) -> Settings:
        return self.settings

    def run(self) -> None:
        """
        Run the pipeline process of continually checking for new experiment files
        and running the experiments sequentially.
        """
        while True:
            # obtain experiment files sorted by creation time
            experiment_files = self.update_experiment_files()

            if len(experiment_files) != 0:
                # obtain the next experiment to process
                next_experiment = Experiment(
                    file_name=experiment_files[0], settings=self.settings
                )

                # proccess the next experiment
                self.process_experiment(experiment=next_experiment)

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
            f"- Average time per query: {round(self.average_processing_time, 3)} seconds"
        )
        logging.info(
            f"- Overall average time per query: {round(self.overall_avg_proc_times, 3)} seconds"
        )

        # log remaining of experiments
        self.update_experiment_files()
        logging.info(f"- Remaining number of experiments: {len(self.experiment_files)}")
        logging.info(f"- Remaining experiments: {self.experiment_files}")

    def process_experiment(
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
        logging.info(f"Sending {len(experiment.number_queries)} queries...")
        asyncio.run(
            self.send_requests_retry(
                remaining_prompts=experiment.experiment_prompts,
            )
        )

        # calculate average processing time per query for the experiment
        end_time = time.time()
        processing_time = end_time - start_time
        avg_query_processing_time = processing_time / experiment.number_queries

        # log completion of experiment
        log_message = (
            f"Completed experiment {experiment}. "
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
        remaining_prompt_dicts: list[dict],
    ) -> None:
        """
        Send requests to the API asynchronously and retry failed queries
        up to a maximum number of attempts.
        """
        # initialise the number of attempts
        attempt = 1

        # send off the requests
        remaining_prompt_dicts, responses = await self.send_requests(
            prompt_dicts=remaining_prompt_dicts,
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
                        prompt_dicts=remaining_prompt_dicts,
                        attempt=attempt,
                    )
                else:
                    # if there are no failed queries, break out of the loop
                    break
            else:
                # if the maximum number of attempts has been reached, break out of the loop
                break
