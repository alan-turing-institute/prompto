import pytest

pytest_plugins = ("pytest_asyncio",)

from prompto.models import AsyncBaseModel
from prompto.settings import Settings


def test_async_base_model_init_errors(temporary_data_folders):
    # not passing in file_name or settings should raise TypeError as they're required
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        AsyncBaseModel()

    # passing in file_name and no settings should raise TypeError as settings is required
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        AsyncBaseModel(settings=Settings())

    # passing in settings and no file_name should raise TypeError as file_name is required
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        AsyncBaseModel(log_file="log_file")


def test_async_base_model_init(temporary_data_folders):
    # intialise settings object for AsyncBaseModel
    settings = Settings()

    # test that the base model class can be instantiated
    async_base_model = AsyncBaseModel(settings=settings, log_file="log_file")
    assert async_base_model.settings == settings
    assert async_base_model.log_file == "log_file"


@pytest.mark.asyncio
async def test_async_base_model_methods(temporary_data_folders):
    # initialise AsyncBaseModel
    async_base_model = AsyncBaseModel(settings=Settings(), log_file="log_file")

    # check .check_environment_variables()
    # raise NotImplementedError as staticmethod call (without instance)
    with pytest.raises(
        NotImplementedError,
        match="'check_environment_variables' method needs to be implemented by a subclass of AsyncBaseModel",
    ):
        AsyncBaseModel.check_environment_variables()
    # raise NotImplementedError as staticmethod call (with instance)
    with pytest.raises(
        NotImplementedError,
        match="'check_environment_variables' method needs to be implemented by a subclass of AsyncBaseModel",
    ):
        async_base_model.check_environment_variables()

    # check .check_prompt_dict()
    # raise NotImplementedError as staticmethod call (without instance)
    with pytest.raises(
        NotImplementedError,
        match="'check_prompt_dict' method needs to be implemented by a subclass of AsyncBaseModel",
    ):
        AsyncBaseModel.check_prompt_dict(prompt_dict={})
    # raise NotImplementedError as staticmethod call (with instance)
    with pytest.raises(
        NotImplementedError,
        match="'check_prompt_dict' method needs to be implemented by a subclass of AsyncBaseModel",
    ):
        async_base_model.check_prompt_dict(prompt_dict={})
    # raises TypeError if called without prompt_dict
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        async_base_model.check_prompt_dict()

    # check .query()
    # should raise AttributeError as AsyncBaseModel does not have query method
    with pytest.raises(
        AttributeError, match="'AsyncBaseModel' object has no attribute 'query'"
    ):
        async_base_model.query(prompt_dict={})

    # check .async_query()
    # raises NotImplementedError as instance method
    with pytest.raises(
        NotImplementedError,
        match="'async_query' method needs to be implemented by a subclass of AsyncBaseModel",
    ):
        await async_base_model.async_query(prompt_dict={})
    # raises TypeError if called without prompt_dict
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        await async_base_model.async_query()
    # raises TypeError if called without instance (as needs self argument)
    with pytest.raises(TypeError, match="missing 2 required positional argument"):
        await AsyncBaseModel.async_query()
