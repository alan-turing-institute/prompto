import os

from prompto.utils import create_folder


class WriteFolderError(Exception):
    pass


class Settings:
    """
    A class to represent the settings for the pipeline/experiment.
    This includes the following attributes:
    - data_folder (folder where input, output, media folders are stored)
    - max_queries (maximum number of queries to send per minute)
    - max_attempts (maximum number of attempts when retrying)
    - parallel (whether to run the experiment(s) in parallel)

    Parameters
    ----------
    data_folder : str, optional
        The folder where the input, output, and media folders are stored, by default "data"
    max_queries : int, optional
        The maximum number of queries to send per minute, by default 10
    max_attempts : int, optional
        The maximum number of attempts when retrying, by default 3
    parallel : bool, optional
        Whether to run the experiment(s) in parallel, by default False
    """

    def __init__(
        self,
        data_folder: str = "data",
        max_queries: int = 10,
        max_attempts: int = 3,
        parallel: bool = False,
    ):
        self._data_folder = data_folder
        # check the data folder exists
        self.check_folder_exists(data_folder)
        # set the subfolders (and create if they do not exist)
        self.set_and_create_subfolders()
        self._max_queries = max_queries
        self._max_attempts = max_attempts
        self.parallel = parallel

    def __str__(self) -> str:
        return (
            f"Settings: data_folder={self.data_folder}, "
            f"max_queries={self.max_queries}, max_attempts={self.max_attempts}, "
            f"parallel={self.parallel}\n"
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
