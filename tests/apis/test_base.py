import pytest

pytest_plugins = ("pytest_asyncio",)

from prompto.apis import AsyncAPI
from prompto.settings import Settings


def test_async_api_init_errors(temporary_data_folders):
    # not passing in file_name or settings should raise TypeError as they're required
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        AsyncAPI()

    # passing in file_name and no settings should raise TypeError as settings is required
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        AsyncAPI(settings=Settings())

    # passing in settings and no file_name should raise TypeError as file_name is required
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        AsyncAPI(log_file="log_file")


def test_async_api_init(temporary_data_folders):
    # initialise settings object for AsyncAPI
    settings = Settings()

    # test that the base model class can be instantiated
    async_api = AsyncAPI(settings=settings, log_file="log_file")
    assert async_api.settings == settings
    assert async_api.log_file == "log_file"


@pytest.mark.asyncio
async def test_async_api_methods(temporary_data_folders):
    # initialise AsyncAPI
    async_api = AsyncAPI(settings=Settings(), log_file="log_file")

    # check .check_environment_variables()
    # raise NotImplementedError as staticmethod call (without instance)
    with pytest.raises(
        NotImplementedError,
        match="'check_environment_variables' method needs to be implemented by a subclass of AsyncAPI",
    ):
        AsyncAPI.check_environment_variables()
    # raise NotImplementedError as staticmethod call (with instance)
    with pytest.raises(
        NotImplementedError,
        match="'check_environment_variables' method needs to be implemented by a subclass of AsyncAPI",
    ):
        async_api.check_environment_variables()

    # check .check_prompt_dict()
    # raise NotImplementedError as staticmethod call (without instance)
    with pytest.raises(
        NotImplementedError,
        match="'check_prompt_dict' method needs to be implemented by a subclass of AsyncAPI",
    ):
        AsyncAPI.check_prompt_dict(prompt_dict={})
    # raise NotImplementedError as staticmethod call (with instance)
    with pytest.raises(
        NotImplementedError,
        match="'check_prompt_dict' method needs to be implemented by a subclass of AsyncAPI",
    ):
        async_api.check_prompt_dict(prompt_dict={})
    # raises TypeError if called without prompt_dict
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        async_api.check_prompt_dict()

    # check .query()
    # raises NotImplementedError as instance method
    with pytest.raises(
        NotImplementedError,
        match="'query' method needs to be implemented by a subclass of AsyncAPI",
    ):
        await async_api.query(prompt_dict={})
    # raises TypeError if called without prompt_dict
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        await async_api.query()
    # raises TypeError if called without instance (as needs self argument)
    with pytest.raises(TypeError, match="missing 2 required positional argument"):
        await AsyncAPI.query()
