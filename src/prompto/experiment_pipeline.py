import asyncio
import logging
from datetime import datetime, timedelta

from prompto.experiment import Experiment
from prompto.settings import Settings
from prompto.utils import sort_jsonl_files_by_creation_time, write_log_message


class ExperimentPipeline:
    """
    A class for the experiment pipeline process.

    Parameters
    ----------
    settings : Settings
        Settings for the pipeline which includes the data folder locations,
        the maximum number of queries to send per minute, the maximum number
        of attempts when retrying, and whether to run the experiment in parallel
    """

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
        Run the pipeline process of continually by checking for
        new experiment files and running the experiments sequentially
        in the order that the files were created.

        The process will continue to run until the program is stopped.
        """
        while True:
            # obtain experiment files sorted by creation/change time
            self.update_experiment_files()

            if len(self.experiment_files) != 0:
                # obtain the next experiment to process
                next_experiment = Experiment(
                    file_name=self.experiment_files[0], settings=self.settings
                )

                # log the estimated time of completion of the next experiment
                self.log_estimate(experiment=next_experiment)

                # proccess the next experiment
                _, avg_query_processing_time = asyncio.run(next_experiment.process())

                # keep track of the average processing time per query for the experiment
                self.average_per_query_processing_times.append(
                    avg_query_processing_time
                )

                # update the overall average processing time per query
                self.overall_avg_proc_times = sum(
                    self.average_per_query_processing_times
                ) / len(self.average_per_query_processing_times)

                # log the progress of the queue of experiments
                self.log_progress(experiment=next_experiment)

    def update_experiment_files(self) -> None:
        """
        Function to update the list of experiment files by sorting
        the files by creation/change time (using `os.path.getctime`).
        """
        self.experiment_files = sort_jsonl_files_by_creation_time(
            input_folder=self.settings.input_folder
        )

    def log_estimate(
        self,
        experiment: Experiment,
    ) -> None:
        """
        Function to log the estimated time of completion of the next experiment.

        Parameters
        ----------
        experiment : Experiment
            The experiment that is being processed
        """
        now = datetime.now()
        if self.overall_avg_proc_times == 0:
            estimated_completion_time = "[unknown]"
            estimated_completion = "[unknown]"
        else:
            estimated_completion_time = round(
                self.overall_avg_proc_times * experiment.number_queries, 3
            )
            estimated_completion = (
                now + timedelta(seconds=estimated_completion_time)
            ).strftime("%d-%m-%Y, %H:%M")

        # log the estimated time of completion of the next experiment
        log_message = (
            f"Next experiment: {experiment}, "
            f"Number of queries: {experiment.number_queries}, "
            f"Estimated completion time: {estimated_completion_time}, "
            f"Estimated completion by: {estimated_completion}"
        )
        write_log_message(log_file=experiment.log_file, log_message=log_message)

    def log_progress(
        self,
        experiment: Experiment,
    ) -> None:
        """
        Function to log the progress of the queue of experiments.

        Parameters
        ----------
        experiment : Experiment
            The experiment that was just processed
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
