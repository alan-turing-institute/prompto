import json
import logging
import os

from tqdm import tqdm


def parse_judge_arg(argument: str) -> list[str]:
    """
    Splits a string into a list by separating on commas.
    Will remove any whitespace and removes duplicates.
    Used to parsing judge argument in CLI commands.

    Parameters
    ----------
    argument : str
        A string of judges separated with commas, e.g.
        "judge1, judge2" or "judge1,judge2,judge1".
        Whitespace will be removed.

    Returns
    -------
    list[str]
        A list of judges with no duplicates, e.g. ["judge1", "judge2"].

    """
    x = argument.replace(" ", "").split(",")
    judges = list(sorted(set(x), key=x.index))
    logging.info(f"Judges to be used: {judges}")
    return judges


def parse_judge_location_arg(judge_location: str) -> tuple[str, dict]:
    """
    Parses the judge location argument to get the
    template prompt string and judge settings dictionary.

    The judge_location argument should be a path to the judge
    folder containing the template.txt and settings.json files.

    We read the template from judge_location/template.txt
    and the settings from judge_location/settings.json. If
    either of these files do not exist, a FileNotFoundError
    will be raised.

    Parameters
    ----------
    judge_location : str
        Path to the judge folder containing the template.txt
        and settings.json files

    Returns
    -------
    tuple[str, dict]
        A tuple containing the template prompt string and
        the judge settings dictionary
    """
    if not os.path.isdir(judge_location):
        raise ValueError(
            f"Judge location '{judge_location}' must be a valid path to a folder"
        )

    try:
        template_path = os.path.join(judge_location, "template.txt")
        with open(template_path, "r", encoding="utf-8") as f:
            template_prompt = f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Template file '{template_path}' does not exist"
        ) from exc

    try:
        judge_settings_path = os.path.join(judge_location, "settings.json")
        with open(judge_settings_path, "r", encoding="utf-8") as f:
            judge_settings = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Settings file '{judge_settings_path}' does not exist"
        ) from exc

    return template_prompt, judge_settings


class Judge:
    """
    Class to create judge inputs for a list of completed responses.

    Parameters
    ----------
    completed_responses : list[dict]
        A list of dictionaries containing the responses to judge.
        Each dictionary should contain the keys "prompt",
        and "response"
    judge_settings : dict
        A dictionary of judge settings with the keys "api",
        "model_name", "parameters". Used to define the
        judge LLMs to be used in the judging process
    template_prompt : str
        A string template to be used to format the prompt
        for the judge LLMs. Often contains placeholders
        for the input prompt (INPUT_PROMPT) and the
        output response (OUTPUT_RESPONSE) which will be formatted
        with the prompt and response from the completed prompt dict
    """

    def __init__(
        self,
        completed_responses: list[dict],
        judge_settings: dict,
        template_prompt: str,
    ):
        self.check_judge_settings(judge_settings)
        self.completed_responses = completed_responses
        self.judge_settings = judge_settings
        self.template_prompt = template_prompt

    @staticmethod
    def check_judge_settings(judge_settings: dict[str, dict]) -> bool:
        """
        Method to check if the judge settings dictionary is valid.

        Parameters
        ----------
        judge_settings : dict
            A dictionary of judge settings with the keys "api",
            "model_name", "parameters". Used to define the
            judge LLMs to be used in the judging process

        Returns
        -------
        bool
            True if the judge settings dictionary is valid.
            Errors will be raised if the dictionary is invalid
        """
        if not isinstance(judge_settings, dict):
            raise TypeError("judge_settings must be a dictionary")

        for judge, settings in judge_settings.items():
            if not isinstance(settings, dict):
                raise TypeError(f"Value for judge key '{judge}' must be a dictionary")
            if "api" not in settings:
                raise KeyError(f"'api' key not found in settings for judge '{judge}'")
            if "model_name" not in settings:
                raise KeyError(
                    f"'model_name' key not found in settings for judge '{judge}'"
                )
            if "parameters" not in settings:
                raise KeyError(
                    f"'parameters' key not found in settings for judge '{judge}'"
                )
            if not isinstance(settings["parameters"], dict):
                raise TypeError(
                    f"Value for 'parameters' key must be a dictionary for judge '{judge}'"
                )

        return True

    @staticmethod
    def check_judge_in_judge_settings(
        judge: str | list[str], judge_settings: dict[str, dict]
    ) -> bool:
        if isinstance(judge, str):
            judge = [judge]

        for j in judge:
            if not isinstance(j, str):
                raise TypeError("If judge is a list, each element must be a string")
            if j not in judge_settings.keys():
                raise KeyError(f"Judge '{j}' is not a key in judge_settings")

        return True

    def create_judge_inputs(self, judge: list[str] | str) -> list[dict]:
        """
        Method to create a list of input prompt dictionaries to
        be processed by the judge LLM(s).

        Parameters
        ----------
        judge : list[str] | str
            A list of judges or a single judge to be used to.
            These must be keys in the judge settings dictionary,
            otherwise a KeyError will be raised

        Returns
        -------
        list[dict]
            A list of dictionaries containing the input prompt
            for the judge LLM(s). Each dictionary will contain a
            new prompt for each prompt/response pair in the
            completed_responses list using the template_prompt
        """
        if isinstance(judge, str):
            judge = [judge]

        self.check_judge_in_judge_settings(judge, self.judge_settings)

        judge_inputs = []
        for j in judge:
            judge_inputs += [
                {
                    "id": f"judge-{str(response.get('id', 'NA'))}",
                    "prompt": self.template_prompt.format(
                        INPUT_PROMPT=response["prompt"],
                        OUTPUT_RESPONSE=response["response"],
                    ),
                    "api": self.judge_settings[judge]["api"],
                    "model_name": self.judge_settings[judge]["model_name"],
                    "parameters": self.judge_settings[judge]["parameters"],
                }
                for response in tqdm(
                    self.completed_responses,
                    desc=f"Creating judge inputs for {j}",
                    unit="responses",
                )
            ]

        return judge_inputs

    def create_judge_file(
        self, judge: list[str] | str, out_filepath: str
    ) -> list[dict]:
        """
        Method to create a judge file (as a jsonl file) containing
        the input prompt dictionaries to be processed by the judge LLM(s).

        Parameters
        ----------
        judge : list[str] | str
            A list of judges or a single judge to be used to.
            These must be keys in the judge settings dictionary,
            otherwise a KeyError will be raised
        out_filepath : str
            The path to the output file where the judge inputs
            will be saved as a jsonl file
        """
        if not out_filepath.endswith(".jsonl"):
            raise ValueError("Output file must be a jsonl file")

        judge_inputs = self.create_judge_inputs(judge)
        with open(out_filepath, "w", encoding="utf-8") as f:
            for j_input in judge_inputs:
                json.dump(j_input, f)
                f.write("\n")

        return judge_inputs
