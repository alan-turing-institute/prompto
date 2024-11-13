import json
import logging
import os

from tqdm import tqdm


def load_rephrase_folder(
    rephrase_folder: str, templates: str = "template.txt"
) -> tuple[list[str], dict]:
    """
    Parses the rephrase_folder to load the template prompt
    string and rephrase settings dictionary.

    The rephrase_folder should be a path to the rephrase
    folder containing the template files and settings.json files.

    We read the template from rephrase_folder/template.txt
    and the settings from rephrase_folder/settings.json. If
    either of these files do not exist, a FileNotFoundError
    will be raised.

    Parameters
    ----------
    rephrase_folder : str
        Path to the rephrase folder containing the template files
        and settings.json files
    templates : str
        Path to the template file to be used for the rephrasals.
        Each line in the template file should contain a template
        for the rephrasal prompt with {INPUT_PROMPT} as a placeholder.
        By default, this is 'template.txt'. This file must be
        in the rephrase folder and end with '.txt'

    Returns
    -------
    tuple[list[str], dict]
        A tuple containing the template prompt string, which
        are given as a list of strings and the rephrase
        settings dictionary
    """
    if not os.path.isdir(rephrase_folder):
        raise ValueError(
            f"rephrase folder '{rephrase_folder}' must be a valid path to a folder"
        )

    template_path = os.path.join(rephrase_folder, templates)
    if not template_path.endswith(".txt"):
        raise ValueError(f"Template file '{template_path}' must end with '.txt'")

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            # reading lines of the template file
            # by default when "\n" is present in the file, it is read as "\\n"
            # so we replace it with "\n"
            template_prompts = [x.replace("\\n", "\n").strip() for x in f.readlines()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Template file '{template_path}' does not exist"
        ) from exc

    try:
        rephrase_settings_path = os.path.join(rephrase_folder, "settings.json")
        with open(rephrase_settings_path, "r", encoding="utf-8") as f:
            rephrase_settings = json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Rephraser settings file '{rephrase_settings_path}' does not exist"
        ) from exc

    return template_prompts, rephrase_settings


class Rephraser:
    """
    Class to create rephrase inputs for a list of input prompts.

    Parameters
    ----------
    input_prompts : list[dict]
        A list of dictionaries containing the input prompt dictionaries
        to rephrase.
    template_prompts : list[str]
        A dictionary containing the template prompt strings
        to be used for the rephrase LLMs. The keys should be the
        name of the template and the value should be the template.
        The string templates (the values) are to be used to format
        the prompt for the rephrase LLMs. Often contains placeholders
        for the input prompt (INPUT_PROMPT) which will be formatted
        with the prompt from the input prompt dict
    rephrase_settings : dict
        A dictionary of rephrase settings with the keys "api",
        "model_name", "parameters". Used to define the
        rephrase LLMs to be used in the judging process
    """

    def __init__(
        self,
        input_prompts: list[dict],
        template_prompts: list[str],
        rephrase_settings: dict,
    ):
        self.check_rephrase_settings(rephrase_settings)
        self.input_prompts = input_prompts
        self.template_prompts = template_prompts
        self.rephrase_settings = rephrase_settings
        self.rephrased_prompts: list[dict] = []

    @staticmethod
    def check_rephrase_settings(rephrase_settings: dict[str, dict]) -> bool:
        """
        Method to check if the rephrase settings dictionary is valid.

        Parameters
        ----------
        rephrase_settings : dict
            A dictionary of rephrase settings with the keys "api",
            "model_name", "parameters". Used to define the
            rephrase LLMs to be used in the judging process

        Returns
        -------
        bool
            True if the rephrase settings dictionary is valid.
            Errors will be raised if the dictionary is invalid
        """
        if not isinstance(rephrase_settings, dict):
            raise TypeError("rephrase_settings must be a dictionary")

        for rephrase, settings in rephrase_settings.items():
            if not isinstance(settings, dict):
                raise TypeError(
                    f"Value for rephrase key '{rephrase}' must be a dictionary"
                )
            if "api" not in settings:
                raise KeyError(
                    f"'api' key not found in settings for rephrase model '{rephrase}'"
                )
            if "model_name" not in settings:
                raise KeyError(
                    f"'model_name' key not found in settings for rephrase model '{rephrase}'"
                )
            if "parameters" not in settings:
                raise KeyError(
                    f"'parameters' key not found in settings for rephrase model '{rephrase}'"
                )
            if not isinstance(settings["parameters"], dict):
                raise TypeError(
                    f"Value for 'parameters' key must be a dictionary for rephrase model '{rephrase}'"
                )

        return True

    @staticmethod
    def check_rephrase_model_in_rephrase_settings(
        rephrase_model: str | list[str], rephrase_settings: dict[str, dict]
    ) -> bool:
        """
        Method to check if the rephrase(s) are in the rephrase settings dictionary.

        Parameters
        ----------
        rephrase_model : str | list[str]
            A list of models or a single model to be used for rephrasals.
            These must be keys in the rephrase settings dictionary,
            otherwise a KeyError will be raised
        rephrase_settings : dict[str, dict]
            A dictionary of rephrase settings with the keys "api",
            "model_name", "parameters". Used to define the
            rephrase LLMs to be used in the judging process

        Returns
        -------
        bool
            True if the rephrase(s) are in the rephrase settings dictionary.
            Errors will be raised if the rephrase(s) are not in the dictionary
        """
        if isinstance(rephrase_model, str):
            rephrase_model = [rephrase_model]

        for j in rephrase_model:
            if not isinstance(j, str):
                raise TypeError(
                    "If rephrase_model is a list, each element must be a string"
                )
            if j not in rephrase_settings.keys():
                raise KeyError(f"Rephraser '{j}' is not a key in rephrase_settings")

        return True

    def create_rephrase_inputs(self, rephrase_model: list[str] | str) -> list[dict]:
        """
        Method to create a list of input prompt dictionaries to
        be processed by the model(s) for rephrasals.

        Parameters
        ----------
        rephrase_model : list[str] | str
            A list of models or a single model to be used for rephrasals.
            These must be keys in the rephrase settings dictionary,
            otherwise a KeyError will be raised

        Returns
        -------
        list[dict]
            A list of dictionaries containing the input prompt
            for the LLM(s) for rephrasal. Each dictionary will contain a
            new prompt for rephrasal for each input prompt in the
            input_prompts list using the template_prompt
        """
        if isinstance(rephrase_model, str):
            rephrase_model = [rephrase_model]

        assert self.check_rephrase_model_in_rephrase_settings(
            rephrase_model=rephrase_model, rephrase_settings=self.rephrase_settings
        )

        self.rephrased_prompts = []
        for r in rephrase_model:
            for i, template_prompt in enumerate(self.template_prompts):
                self.rephrased_prompts += [
                    {
                        "id": f"rephrase-{r}-{i}-{str(input.get('id', 'NA'))}",
                        "template_index": i,
                        "prompt": template_prompt.format(
                            INPUT_PROMPT=input["prompt"],
                        ),
                    }
                    | self.rephrase_settings[r]
                    | {f"input-{k}": v for k, v in input.items()}
                    for input in tqdm(
                        self.input_prompts,
                        desc=f"Creating rephrase inputs for rephrase model '{r}' and template '{i}'",
                        unit="inputs",
                    )
                ]

        return self.rephrased_prompts

    def create_rephrase_file(
        self, rephrase_model: list[str] | str, out_filepath: str
    ) -> list[dict]:
        """
        Method to create a rephrase file (as a jsonl file) containing
        the input prompt dictionaries to be processed by the model(s) for rephrasals.

        Parameters
        ----------
        rephrase_model : list[str] | str
            A list of models or a single model to be used for rephrasals.
            These must be keys in the rephrase settings dictionary,
            otherwise a KeyError will be raised
        out_filepath : str
            The path to the output file where the rephrase inputs
            will be saved as a jsonl file

        Returns
        -------
        list[dict]
            A list of dictionaries containing the input prompt
            for the LLM(s) for rephrasal. Each dictionary will contain a
            new prompt for rephrasal for each input prompt in the
            input_prompts list using the template_prompt
        """
        if not out_filepath.endswith(".jsonl"):
            raise ValueError("out_filepath must end with '.jsonl'")

        rephrased_prompts = self.create_rephrase_inputs(rephrase_model=rephrase_model)

        logging.info(f"Creating rephrase experiment file at {out_filepath}...")
        with open(out_filepath, "w", encoding="utf-8") as f:
            for j_input in tqdm(
                rephrased_prompts,
                desc=f"Writing rephrase prompts to {out_filepath}",
                unit="prompts",
            ):
                json.dump(j_input, f)
                f.write("\n")

        return rephrased_prompts

    @staticmethod
    def _convert_rephrased_prompt_dict_to_input(rephrased_prompt: dict) -> dict:
        """
        Method to convert a completed rephrased prompt dictionary to an input prompt dictionary.
        This is done by:
        - Renaming the "response" key to "prompt" (as this is the new rephrased prompt)
        - Keep "id" key as is
        - Keep "input-prompt" and "input-id" keys as is
        - For all remaining keys starting with "input-", remove the "input-" prefix.
          This may overwrite existing keys in the rephrased prompt dictionary
          (e.g. "input-api", "input-model_name", "input-parameters"
          should override existing keys "api", "model_name", "parameters")

        Parameters
        ----------
        rephrased_prompt : dict
            A dictionary containing the rephrased prompt. Should usually contain
            the keys "id", "prompt", "input-prompt" and "input-id". Should also
            contain "input-api", "input-model_name" and "input-parameters" keys

        Returns
        -------
        dict
            A dictionary containing the input prompt for a model after rephrasing.
            The prompt will be the rephrased prompt, and there will be "input-id"
            and "input-prompt" keys to keep track of the original input prompt.
            The "id" key will indicate the rephrased prompt id. The "api", "model_name",
            and other keys from the original input will be restored
        """
        input_prompt = {
            "id": rephrased_prompt["id"],
            "prompt": rephrased_prompt["prompt"],
            "input-prompt": rephrased_prompt["input-prompt"],
            "input-id": rephrased_prompt.get("input-id"),
        }

        # restore the original input keys (e.g. "api", "model_name", "parameters")
        for k, v in rephrased_prompt.items():
            if k.startswith("input-") and k not in ["input-prompt", "input-id"]:
                input_prompt[k[6:]] = v

        return input_prompt

    def create_new_input_file(
        self,
        keep_original: bool,
        completed_rephrase_responses: list[dict],
        out_filepath: str,
    ) -> list[dict]:
        """
        Method to create a new input file given the original input prompts and
        the completed rephrase responses. This is done by matching the "input-id"
        key in the rephrase responses with the "id" key in the input prompts.

        There is an option to keep the original input prompts, or to remove them (i.e.
        only keep the rephrased prompts).

        Parameters
        ----------
        keep_original : bool
            Whether or not to keep the original input prompts in the new input file.
            If True, the original input prompts will be kept in the new input file
        completed_rephrase_responses : list[dict]
            A list of dictionaries containing the completed rephrased prompts.
            Each dictionary should usually contain the keys "id", "prompt",
            "input-prompt" and "input-id". They should also contain "input-api",
            "input-model_name" and "input-parameters" keys
        out_filepath : str
            The path to the output file where the new input prompts will
            be saved as a jsonl file

        Returns
        -------
        list[dict]
            A list of dictionaries containing the input prompts for the models
            after rephrasing. The prompt will be the rephrased prompt, and there
            will be "input-id" and "input-prompt" keys to keep track of the original
            input prompt. The "id" key will indicate the rephrased prompt id. The "api",
            "model_name", and other keys from the original input will be restored
        """
        if not out_filepath.endswith(".jsonl"):
            raise ValueError("out_filepath must end with '.jsonl'")

        # obtain the new rephrased prompts
        new_input_prompts = [
            self._convert_rephrased_prompt_dict_to_input(rephrased_prompt)
            for rephrased_prompt in completed_rephrase_responses
        ]

        # add the original input prompts if keep_original is True
        if keep_original:
            new_input_prompts += [
                x | {"input-id": x.get("id")} for x in self.input_prompts
            ]

        logging.info(
            f"Creating new input file with rephrased prompts at {out_filepath}..."
        )
        with open(out_filepath, "w", encoding="utf-8") as f:
            for j_input in tqdm(
                new_input_prompts,
                desc=f"Writing new input prompts to {out_filepath}",
                unit="prompts",
            ):
                json.dump(j_input, f)
                f.write("\n")

        return new_input_prompts
