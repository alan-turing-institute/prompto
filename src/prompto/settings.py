import os

from prompto.utils import create_folder


class WriteFolderError(Exception):
    pass


def check_max_queries_dict(max_queries_dict: dict[str, int | dict[str, int]]) -> None:
    """
    Check the format of the max_queries_dict dictionary.

    Raises a ValueError if the dictionary is not in the correct format.

    Parameters
    ----------
    max_queries_dict : dict[str, int | dict[str, int]], optional
        A dictionary of maximum queries per minute for each API or group, by default {}.
        The dictionary keys should be either a group name (which is then used in the
        "group" key of the prompt_dict) or an API name. The values should be integers
        (the maximum queries per minute or rate limit) or itself a dictionary with
        keys as the model-names and values as the maximum queries per minute for that model.
    """
    # check max_queries_dict is a dictionary
    if not isinstance(max_queries_dict, dict):
        raise ValueError(
            f"max_queries_dict must be a dictionary, not {type(max_queries_dict)}"
        )

    for key, value in max_queries_dict.items():
        # check each key is a string
        if not isinstance(key, str):
            raise ValueError(f"max_queries_dict keys must be strings, not {type(key)}")

        # check each value is an integer or dictionary
        if not isinstance(value, int) and not isinstance(value, dict):
            raise ValueError(
                f"max_queries_dict values must be integers or dictionaries, not {type(value)}"
            )

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # check each sub_key is a string
                if not isinstance(sub_key, str):
                    raise ValueError(
                        "if a value of max_queries_dict is a dictionary, "
                        f"the sub-keys must be strings, not {type(sub_key)}"
                    )

                # check each sub_value is an integer
                if not isinstance(sub_value, int):
                    raise ValueError(
                        "if a value of max_queries_dict is a dictionary, "
                        f"sub-values must be integers, not {type(sub_value)}"
                    )


class Settings:
    """
    A class to represent the settings for the pipeline/experiment.
    This includes the following attributes:
    - data_folder (folder where input, output, media folders are stored)
    - max_queries (default maximum number of queries to send per minute)
    - max_attempts (maximum number of attempts when retrying)
    - parallel (whether to run the experiment(s) in parallel)
    - max_queries_dict (dictionary of maximum queries per minute for each API or group)

    Parameters
    ----------
    data_folder : str, optional
        The folder where the input, output, and media folders are stored, by default "data"
    max_queries : int, optional
        The default maximum number of queries to send per minute, by default 10
    max_attempts : int, optional
        The maximum number of attempts when retrying, by default 3
    parallel : bool, optional
        Whether to run the experiment(s) in parallel, by default False
    max_queries_dict : dict[str, int | dict[str, int]], optional
        A dictionary of maximum queries per minute for each API or group, by default {}.
        The dictionary keys should be either a group name (which is then used in the
        "group" key of the prompt_dict) or an API name. The values should be integers
        (the maximum queries per minute or rate limit) or itself a dictionary with
        keys as the model-names and values as the maximum queries per minute for that model.
    """

    def __init__(
        self,
        data_folder: str = "data",
        max_queries: int = 10,
        max_attempts: int = 3,
        parallel: bool = False,
        max_queries_dict: dict[str, int | dict[str, int]] = {},
    ):
        # check the data folder exists
        self.check_folder_exists(data_folder)
        # check form of max_queries_dict
        check_max_queries_dict(max_queries_dict)

        self._data_folder = data_folder
        # set the subfolders (and create if they do not exist)
        self.set_and_create_subfolders()
        self._max_queries = max_queries
        self._max_attempts = max_attempts
        # set parallel settings
        self.parallel = parallel
        self.max_queries_dict = max_queries_dict

    def __str__(self) -> str:
        return (
            f"Settings: data_folder={self.data_folder}, "
            f"max_queries={self.max_queries}, max_attempts={self.max_attempts}, "
            f"parallel={self.parallel}\n"
            f"max_queries_dict={self.max_queries_dict}\n"
            f"Subfolders: input_folder={self.input_folder}, "
            f"output_folder={self.output_folder}, media_folder={self.media_folder}"
        )

    @staticmethod
    def check_folder_exists(data_folder: str) -> bool:
        """
        Check that the data folder exists.

        Raises a ValueError if the data folder does not exist.

        Parameters
        ----------
        data_folder : str
            The path to the data folder

        Returns
        -------
        bool
            True if the data folder exists, otherwise raises a ValueError
        """
        # check if data folder exists
        if not os.path.isdir(data_folder):
            raise ValueError(
                f"Data folder '{data_folder}' must be a valid path to a folder"
            )

        return True

    def set_subfolders(self) -> None:
        """
        Set the subfolders for the data folder.

        The subfolders are:
        - input_folder: folder where input data is stored (e.g. experiment files)
        - output_folder: folder where output data is stored (e.g. results, logs)
        - media_folder: folder where media files are stored (e.g. images, videos)

        They are stored in the data folder.
        """
        self._input_folder = os.path.join(self.data_folder, "input")
        self._output_folder = os.path.join(self.data_folder, "output")
        self._media_folder = os.path.join(self.data_folder, "media")

    def create_subfolders(self) -> None:
        """
        Create the subfolders for the data folder.

        The subfolders must be set before calling this method.
        """
        # check all folders exist and create them if not
        for folder in [self._input_folder, self._output_folder, self._media_folder]:
            create_folder(folder)

    def set_and_create_subfolders(self) -> None:
        """
        Set and create the subfolders for the data folder.
        """
        # set the subfolders and create them if they do not exist
        self.set_subfolders()
        self.create_subfolders()

    # ---- data folder ----

    @property
    def data_folder(self) -> str:
        return self._data_folder

    @data_folder.setter
    def data_folder(self, value: str):
        # check the data folder exists
        self.check_folder_exists(value)
        # set the data folder
        self._data_folder = value
        # set and create any subfolders if they do not exist
        self.set_and_create_subfolders()

    # ---- input folder (read only) ----

    @property
    def input_folder(self) -> str:
        return self._input_folder

    @input_folder.setter
    def input_folder(self, value: str):
        raise WriteFolderError(
            "Cannot set input folder on it's own. Set the 'data_folder' instead"
        )

    # ---- output folder (read only) ----

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value: str):
        raise WriteFolderError(
            "Cannot set output folder on it's own. Set the 'data_folder' instead"
        )

    # ---- media folder (read only) ----

    @property
    def media_folder(self) -> str:
        return self._media_folder

    @media_folder.setter
    def media_folder(self, value: str):
        raise WriteFolderError(
            "Cannot set media folder on it's own. Set the 'data_folder' instead"
        )

    # ---- max queries ----

    @property
    def max_queries(self) -> int:
        return self._max_queries

    @max_queries.setter
    def max_queries(self, value: int):
        self._max_queries = value

    # ---- max attempts ----

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @max_attempts.setter
    def max_attempts(self, value: int):
        self._max_attempts = value
