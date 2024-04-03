import os

from batch_llm.file_operations import create_folder


class WriteFolderError(Exception):
    pass


class Settings:
    # settings for the pipeline which includes the following attributes:
    # - data_folder (folder where input, output, media folders are stored)
    # - max_queries (maximum number of queries to send per minute)
    # - max_attempts (maximum number of attempts when retrying)

    def __init__(
        self, data_folder: str = "data", max_queries: int = 10, max_attempts: int = 3
    ):
        self._data_folder = data_folder
        # check the data folder exists
        self.check_folder_exists(data_folder)
        # set the subfolders (and create if they do not exist)
        self.set_subfolders()
        self._max_queries = max_queries
        self._max_attempts = max_attempts

    @classmethod
    def check_folder_exists(data_folder: str) -> tuple[str]:
        """
        Check that the data folder exists.

        Raises a ValueError if the data folder does not exist.
        """
        # check if data folder exists
        if not os.path.exists(data_folder):
            raise ValueError(f"Data folder {data_folder} does not exist.")

    def set_subfolders(self) -> None:
        # set the subfolders for the data folder
        self._input_folder = os.path.join(self.data_folder, "input")
        self._output_folder = os.path.join(self.data_folder, "output")
        self._media_folder = os.path.join(self.data_folder, "media")

    def create_subfolders(self) -> None:
        # check all folders exist and create them if not
        for folder in [self._input_folder, self._output_folder, self._media_folder]:
            create_folder(folder)

    def set_and_create_subfolders(self) -> None:
        # set the subfolders and create them if they do not exist
        self.set_subfolders()
        self.create_subfolders()

    # ---- data folder ----

    @property
    def data_folder(self) -> str:
        return self.data_folder

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
            "Cannot write to input folder on it's own. Use the 'set_and_create_subfolders' method to set the input folder."
        )

    # ---- output folder (read only) ----

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @output_folder.setter
    def output_folder(self, value: str):
        raise WriteFolderError(
            "Cannot write to output folder on it's own. Use the 'set_and_create_subfolders' method to set the output folder."
        )

    # ---- media folder (read only) ----

    @property
    def media_folder(self) -> str:
        return self._media_folder

    @media_folder.setter
    def media_folder(self, value: str):
        raise WriteFolderError(
            "Cannot write to media folder on it's own. Use the 'set_and_create_subfolders' method to set the media folder."
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
