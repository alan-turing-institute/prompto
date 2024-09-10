import json
import os

from tqdm import tqdm


def load_judge_folder(
    judge_folder: str, templates: str | list[str] = "template.txt"
) -> tuple[dict[str, str], dict]:
    """
    Parses the judge_folder to load the template prompt
    string and judge settings dictionary.

    ≈ should be a path to the judge
    folder containing the template.txt and settings.json files.

    We read the template from judge_folder/template.txt
    and the settings from judge_folder/settings.json. If
    either of these files do not exist, a FileNotFoundError
    will be raised.

    Parameters
    ----------
    judge_folder : str
        Path to the judge folder containing the template.txt
        and settings.json files
    templates : str | list[str]
        Path(s) to the template file(s) to be used for the judge.
        By default, this is 'template.txt'. These files must be
        in the judge folder and end with '.txt'

    Returns
    -------
    tuple[dict[str, str], dict]
        A tuple containing the template prompt string, which
        are given as a dictionary with the template name as the
        key (the template file name without the '.txt' extension)
        and the value as the template string, and the judge
        settings dictionary
    """
    if not os.path.isdir(judge_folder):
        raise ValueError(
            f"judge folder '{judge_folder}' must be a valid path to a folder"
        )
    if isinstance(templates, str):
        templates = [templates]

    template_prompts = {}
    for template in templates:
        template_path = os.path.join(judge_folder, template)
        if not template_path.endswith(".txt"):
            raise ValueError(f"Template file '{template_path}' must end with '.txt'")

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template_prompts[template.split(".")[0]] = f.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Template file '{template_path}' does not exist"
            ) from exc

    try:
        judge_settings_path = os.path.join(judge_folder, "settings.json")
        with open(judge_settings_path, "r", encoding="utf-8") as f:
            judge_settings = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Judge settings file '{judge_settings_path}' does not exist"
        ) from exc

    return template_prompts, judge_settings


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
    template_prompt : dict[str, str]
        A dictionary containing the template prompt strings
        to be used for the judge LLMs. The keys should be the
        name of the template and the value should be the template.
        The string templates (the values) are to be used to format
        the prompt for the judge LLMs. Often contains placeholders
        for the input prompt (INPUT_PROMPT) and the
        output response (OUTPUT_RESPONSE) which will be formatted
        with the prompt and response from the completed prompt dict
    """

    def __init__(
        self,
        completed_responses: list[dict],
        judge_settings: dict,
        template_prompts: dict[str, str],
    ):
        self.check_judge_settings(judge_settings)
        if not isinstance(template_prompts, dict):
            raise TypeError("template_prompts must be a dictionary")
        self.completed_responses = completed_responses
        self.judge_settings = judge_settings
        self.template_prompts = template_prompts

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
        """
        Method to check if the judge(s) are in the judge settings dictionary.

        Parameters
        ----------
        judge : str | list[str]
            A single judge or a list of judges to check if they
            are keys in the judge_settings dictionary
        judge_settings : dict[str, dict]
            A dictionary of judge settings with the keys "api",
            "model_name", "parameters". Used to define the
            judge LLMs to be used in the judging process

        Returns
        -------
        bool
            True if the judge(s) are in the judge settings dictionary.
            Errors will be raised if the judge(s) are not in the dictionary
        """
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

        assert self.check_judge_in_judge_settings(judge, self.judge_settings)

        judge_inputs = []
        for j in judge:
            for template_name, template_prompt in self.template_prompts.items():
                judge_inputs += [
                    {
                        "id": f"judge-{j}-{template_name}-{str(response.get('id', 'NA'))}",
                        "template_name": template_name,
                        "prompt": template_prompt.format(
                            INPUT_PROMPT=response["prompt"],
                            OUTPUT_RESPONSE=response["response"],
                        ),
                        "api": self.judge_settings[j]["api"],
                        "model_name": self.judge_settings[j]["model_name"],
                        "parameters": self.judge_settings[j]["parameters"],
                    }
                    | {f"input-{k}": v for k, v in response.items()}
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
            raise ValueError("out_filepath must end with '.jsonl'")

        judge_inputs = self.create_judge_inputs(judge=judge)
        with open(out_filepath, "w", encoding="utf-8") as f:
            for j_input in judge_inputs:
                json.dump(j_input, f)
                f.write("\n")

        return judge_inputs
