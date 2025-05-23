import logging
import os

import pytest

from prompto.settings import Settings, WriteFolderError


def test_settings_default_init(temporary_data_folders):
    settings = Settings()

    # check the default values
    assert settings.data_folder == "data"
    assert settings.max_queries == 10
    assert settings.max_attempts == 3
    assert settings.parallel is False
    assert settings.max_queries_dict == {}

    # check the subfolders
    assert settings.input_folder == "data/input"
    assert settings.output_folder == "data/output"
    assert settings.media_folder == "data/media"

    # check the folders exist (should be created by initialising settings object)
    assert os.path.isdir("data/input")
    assert os.path.isdir("data/output")
    assert os.path.isdir("data/media")

    # when printing, it should show the settings and subfolders
    assert str(settings) == (
        "Settings: data_folder=data, max_queries=10, max_attempts=3, parallel=False\n"
        "Subfolders: input_folder=data/input, output_folder=data/output, media_folder=data/media"
    )


def test_settings_init_errors(temporary_data_folders):
    # raise an error if max_queries is not an integer
    with pytest.raises(TypeError, match="max_queries must be a positive integer"):
        Settings(max_queries="not_an_int")

    # raise an error if max_queries is a negative integer
    with pytest.raises(ValueError, match="max_queries must be a positive integer"):
        Settings(max_queries=0)

    # raise an error if max_attempts is not an integer
    with pytest.raises(TypeError, match="max_attempts must be a positive integer"):
        Settings(max_attempts="not_an_int")

    # raise an error if max_attempts is a negative integer
    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        Settings(max_attempts=0)


def test_settings_custom_init(temporary_data_folders):
    settings = Settings(
        data_folder="dummy_data", max_queries=20, max_attempts=5, parallel=True
    )

    # check the custom values
    assert settings.data_folder == "dummy_data"
    assert settings.max_queries == 20
    assert settings.max_attempts == 5
    assert settings.parallel is True
    assert settings.max_queries_dict == {}

    # check the subfolders
    assert settings.input_folder == "dummy_data/input"
    assert settings.output_folder == "dummy_data/output"
    assert settings.media_folder == "dummy_data/media"

    # check the folders exist (should be created by initialising settings object)
    assert os.path.isdir("dummy_data/input")
    assert os.path.isdir("dummy_data/output")
    assert os.path.isdir("dummy_data/media")

    # when printing with parallel=True, should print out the max_queries_dict
    assert str(settings) == (
        "Settings: data_folder=dummy_data, max_queries=20, max_attempts=5, parallel=True, "
        "max_queries_dict={}\n"
        "Subfolders: input_folder=dummy_data/input, "
        "output_folder=dummy_data/output, "
        "media_folder=dummy_data/media"
    )


def test_settings_custom_init_with_max_queries_dict(temporary_data_folders):
    settings = Settings(
        data_folder="dummy_data",
        max_queries=20,
        max_attempts=5,
        parallel=True,
        max_queries_dict={"api1": 10, "api2": {"model1": 10, "model2": 20}},
    )

    # check the custom values
    assert settings.data_folder == "dummy_data"
    assert settings.max_queries == 20
    assert settings.max_attempts == 5
    assert settings.parallel is True
    assert settings.max_queries_dict == {
        "api1": 10,
        "api2": {"model1": 10, "model2": 20},
    }

    # check the subfolders
    assert settings.input_folder == "dummy_data/input"
    assert settings.output_folder == "dummy_data/output"
    assert settings.media_folder == "dummy_data/media"

    # check the folders exist (should be created by initialising settings object)
    assert os.path.isdir("dummy_data/input")
    assert os.path.isdir("dummy_data/output")
    assert os.path.isdir("dummy_data/media")

    # when printing with parallel=True, should print out the max_queries_dict
    assert str(settings) == (
        "Settings: data_folder=dummy_data, max_queries=20, max_attempts=5, parallel=True, "
        "max_queries_dict={'api1': 10, 'api2': {'model1': 10, 'model2': 20}}\n"
        "Subfolders: input_folder=dummy_data/input, "
        "output_folder=dummy_data/output, "
        "media_folder=dummy_data/media"
    )


def test_settings_custom_init_with_max_queries_dict_and_no_parallel(
    temporary_data_folders,
):
    settings = Settings(
        data_folder="dummy_data",
        max_queries=20,
        max_attempts=5,
        parallel=False,
        max_queries_dict={"api1": 10, "api2": {"model1": 10, "model2": 20}},
    )

    # check the custom values
    assert settings.data_folder == "dummy_data"
    assert settings.max_queries == 20
    assert settings.max_attempts == 5
    assert settings.parallel is False
    assert settings.max_queries_dict == {
        "api1": 10,
        "api2": {"model1": 10, "model2": 20},
    }

    # check the subfolders
    assert settings.input_folder == "dummy_data/input"
    assert settings.output_folder == "dummy_data/output"
    assert settings.media_folder == "dummy_data/media"

    # check the folders exist (should be created by initialising settings object)
    assert os.path.isdir("dummy_data/input")
    assert os.path.isdir("dummy_data/output")
    assert os.path.isdir("dummy_data/media")

    # when printing with parallel=True, should print out the max_queries_dict
    assert str(settings) == (
        "Settings: data_folder=dummy_data, max_queries=20, max_attempts=5, parallel=False\n"
        "Subfolders: input_folder=dummy_data/input, "
        "output_folder=dummy_data/output, "
        "media_folder=dummy_data/media"
    )


def test_warning_max_queries_dict_not_used(temporary_data_folders, caplog):
    caplog.set_level(logging.WARNING)

    Settings(
        data_folder="dummy_data",
        max_queries=20,
        max_attempts=5,
        parallel=False,
        max_queries_dict={"api1": 10, "api2": {"model1": 10, "model2": 20}},
    )

    # check the warning message
    log_msg = (
        "max_queries_dict is provided and not empty, but parallel is set to False, so "
        "max_queries_dict will not be used. Set parallel to True to use max_queries_dict"
    )
    assert log_msg in caplog.text


def test_settings_check_max_queries_dict_errors(temporary_data_folders):
    # check raise error if max_queries_dict is not a dictionary
    max_queries_dict = "not_a_dict"
    with pytest.raises(
        TypeError,
        match="max_queries_dict must be a dictionary, not <class 'str'>",
    ):
        Settings(max_queries_dict=max_queries_dict)

    # check raise error if there is a key that is not a string
    max_queries_dict = {"api": 10, 1: 10}
    with pytest.raises(
        TypeError,
        match="max_queries_dict keys must be strings, not <class 'int'>",
    ):
        Settings(max_queries_dict=max_queries_dict)

    # check raise error if there is a value that is not an integer or a dictionary
    max_queries_dict = {"api": 10, "api_2": "not_an_int"}
    with pytest.raises(
        TypeError,
        match="max_queries_dict values must be integers or dictionaries, not <class 'str'>",
    ):
        Settings(max_queries_dict=max_queries_dict)

    # check raise error if there is a value that is a negative integer
    max_queries_dict = {"api": 10, "api_2": -10}
    with pytest.raises(
        ValueError,
        match=(
            "if a value of max_queries_dict is an integer, "
            "the value must be a positive integer, not negative"
        ),
    ):
        Settings(max_queries_dict=max_queries_dict)

    # check raise error if there is a value that is a dictionary
    # which has a key that is not a string
    max_queries_dict = {"api": 10, "api_2": {"model_name": 10, 1: 10}}
    with pytest.raises(
        TypeError,
        match=(
            "if a value of max_queries_dict is a dictionary, "
            "the sub-keys must be strings, not <class 'int'>"
        ),
    ):
        Settings(max_queries_dict=max_queries_dict)

    # check raise error if there is a value that is a dictionary
    # which has a value that is not an integer
    max_queries_dict = {
        "api": 10,
        "api_2": {"model_name": 20, "model_name_2": "not_an_int"},
    }
    with pytest.raises(
        TypeError,
        match=(
            "if a value of max_queries_dict is a dictionary, "
            "the sub-values must be integers, not <class 'str'>"
        ),
    ):
        Settings(max_queries_dict=max_queries_dict)

    # check raise error if there is a value that is a dictionary
    # which has a value that is a negative integer
    max_queries_dict = {"api": 10, "api_2": {"model_name": 20, "model_name_2": -10}}
    with pytest.raises(
        ValueError,
        match=(
            "if a value of max_queries_dict is a dictionary, "
            "the sub-values must be positive integers, not negative"
        ),
    ):
        Settings(max_queries_dict=max_queries_dict)


def test_settings_check_folder_exists(temporary_data_folders):
    # call static method directly
    Settings.check_folder_exists("dummy_data")

    # should raise a ValueError if the path does not exist
    unknown_folder = "unknown_folder"
    with pytest.raises(
        ValueError,
        match=f"Data folder '{unknown_folder}' must be a valid path to a folder",
    ):
        Settings.check_folder_exists(unknown_folder)

    # should raise a ValueError if the path is not a folder/directory
    file_path = "test.txt"
    with pytest.raises(
        ValueError, match=f"Data folder '{file_path}' must be a valid path to a folder"
    ):
        Settings.check_folder_exists(file_path)


def test_settings_set_subfolders(temporary_data_folders):
    settings = Settings()

    # set it to a different folder
    # (manually without triggering the data_folder setter)
    settings._data_folder = "dummy_data"
    settings.set_subfolders()

    # check the subfolders have been set
    assert settings.input_folder == "dummy_data/input"
    assert settings.output_folder == "dummy_data/output"
    assert settings.media_folder == "dummy_data/media"

    # check the folders do not exist yet
    assert not os.path.isdir("dummy_data/input")
    assert not os.path.isdir("dummy_data/output")
    assert not os.path.isdir("dummy_data/media")


def test_settings_create_subfolders(temporary_data_folders):
    settings = Settings()

    # set it to a different folder
    # (manually without triggering the data_folder setter)
    settings._data_folder = "dummy_data"
    settings.set_subfolders()
    settings.create_subfolders()

    # check the folders exist
    assert os.path.isdir("dummy_data/input")
    assert os.path.isdir("dummy_data/output")
    assert os.path.isdir("dummy_data/media")


def test_settings_set_and_create_subfolders(temporary_data_folders):
    settings = Settings()

    # set it to a different folder
    # (manually without triggering the data_folder setter)
    settings._data_folder = "dummy_data"
    settings.set_and_create_subfolders()

    # check the subfolders have been set
    assert settings.input_folder == "dummy_data/input"
    assert settings.output_folder == "dummy_data/output"
    assert settings.media_folder == "dummy_data/media"

    # check the folders exist
    assert os.path.isdir("dummy_data/input")
    assert os.path.isdir("dummy_data/output")
    assert os.path.isdir("dummy_data/media")


def test_settings_data_folder_getter(temporary_data_folders):
    settings = Settings()
    assert settings.data_folder == "data"

    # set it to a different folder
    settings._data_folder = "dummy_data"
    assert settings.data_folder == "dummy_data"


def test_settings_data_folder_setter(temporary_data_folders):
    settings = Settings()

    # set it to a different folder
    # should trigger the subfolders to be set and created
    settings.data_folder = "dummy_data"
    assert settings.data_folder == "dummy_data"

    # check the subfolders have been set
    assert settings.input_folder == "dummy_data/input"
    assert settings.output_folder == "dummy_data/output"
    assert settings.media_folder == "dummy_data/media"

    # check the folders exist
    assert os.path.isdir("dummy_data/input")
    assert os.path.isdir("dummy_data/output")
    assert os.path.isdir("dummy_data/media")


def test_settings_input_folder_getter(temporary_data_folders):
    settings = Settings()
    assert settings.input_folder == "data/input"

    # set it to a different folder
    # (manually circumventing the setter error)
    settings._input_folder = "dummy_data/input"
    assert settings.input_folder == "dummy_data/input"


def test_settings_input_folder_setter(temporary_data_folders):
    settings = Settings()

    # should raise an error if trying to set it directly
    # (should use the data_folder setter)
    with pytest.raises(
        WriteFolderError,
        match="Cannot set input folder on it's own. Set the 'data_folder' instead",
    ):
        settings.input_folder = "dummy_data/input"


def test_settings_output_folder_getter(temporary_data_folders):
    settings = Settings()
    assert settings.output_folder == "data/output"

    # set it to a different folder
    # (manually circumventing the setter error)
    settings._output_folder = "dummy_data/output"
    assert settings.output_folder == "dummy_data/output"


def test_settings_output_folder_setter(temporary_data_folders):
    settings = Settings()

    # should raise an error if trying to set it directly
    # (should use the data_folder setter)
    with pytest.raises(
        WriteFolderError,
        match="Cannot set output folder on it's own. Set the 'data_folder' instead",
    ):
        settings.output_folder = "dummy_data/output"


def test_settings_media_folder_getter(temporary_data_folders):
    settings = Settings()
    assert settings.media_folder == "data/media"

    # set it to a different folder
    # (manually circumventing the setter error)
    settings._media_folder = "dummy_data/media"
    assert settings.media_folder == "dummy_data/media"


def test_settings_media_folder_setter(temporary_data_folders):
    settings = Settings()

    # should raise an error if trying to set it directly
    # (should use the data_folder setter)
    with pytest.raises(
        WriteFolderError,
        match="Cannot set media folder on it's own. Set the 'data_folder' instead",
    ):
        settings.media_folder = "dummy_data/media"


def test_max_queries_getter(temporary_data_folders):
    settings = Settings()

    # check the default value
    assert settings.max_queries == 10

    # set it to a different value
    settings._max_queries = 20
    assert settings.max_queries == 20


def test_max_queries_setter(temporary_data_folders):
    settings = Settings()

    # set it to a different value
    settings.max_queries = 20
    assert settings.max_queries == 20

    # raise an error if setting it to a negative integer
    with pytest.raises(ValueError, match="max_queries must be a positive integer"):
        settings.max_queries = -10

    # raise an error if setting it to 0
    with pytest.raises(ValueError, match="max_queries must be a positive integer"):
        settings.max_queries = 0

    # raise an error if setting it to a non-integer
    with pytest.raises(ValueError, match="max_queries must be a positive integer"):
        settings.max_queries = "not_an_int"


def test_max_attempts_getter(temporary_data_folders):
    settings = Settings()

    # check the default value
    assert settings.max_attempts == 3

    # set it to a different value
    settings._max_attempts = 5
    assert settings.max_attempts == 5


def test_max_attempts_setter(temporary_data_folders):
    settings = Settings()

    # set it to a different value
    settings.max_attempts = 5
    assert settings.max_attempts == 5

    # raise an error if setting it to a negative integer
    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        settings.max_attempts = -10

    # raise an error if setting it to 0
    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        settings.max_attempts = 0

    # raise an error if setting it to a non-integer
    with pytest.raises(ValueError, match="max_attempts must be a positive integer"):
        settings.max_attempts = "not_an_int"


def test_parallel_getter_and_setter(temporary_data_folders):
    settings = Settings()

    # check the default value
    assert settings.parallel is False

    # set it to a different value
    settings.parallel = True
    assert settings.parallel is True


def test_max_queries_dict_getter_and_setter(temporary_data_folders):
    settings = Settings()

    # check the default value
    assert settings.max_queries_dict == {}

    # set it to a different value
    settings.max_queries_dict = {"api1": 10, "api2": {"model1": 10, "model2": 20}}
    assert settings.max_queries_dict == {
        "api1": 10,
        "api2": {"model1": 10, "model2": 20},
    }
