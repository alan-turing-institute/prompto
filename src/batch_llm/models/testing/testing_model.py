import asyncio
import random
import time
from typing import Any

from batch_llm.models.base import AsyncBaseModel, BaseModel
from batch_llm.settings import Settings
from batch_llm.utils import log_error_response_query, log_success_response_query


class TestModel(BaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        return []

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        return []

    def query(prompt_dict: dict, index: int | str) -> dict:
        # return an error 1/5 times
        raise_error = random.randint(1, 5) == 1
        if raise_error:
            error_msg = "This is a test error which we should handle and return"
            log_error_response_query(
                index=index,
                model="test",
                prompt=prompt_dict["prompt"],
                error_as_string=error_msg,
            )
            raise ValueError(error_msg)
        else:
            # wait 15 seconds to simulate a running task
            time.sleep(15)

        response_text = "This is a test response"
        log_success_response_query(
            index=index,
            model="test",
            prompt=prompt_dict["prompt"],
            response_text=response_text,
        )

        prompt_dict["response"] = response_text
        return prompt_dict


class AsyncTestModel(AsyncBaseModel):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        return []

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        return []

    async def async_query(prompt_dict: dict, index: int | str) -> dict:
        # return an error 1/5 times
        raise_error = random.randint(1, 5) == 1
        if raise_error:
            error_msg = "This is a test error which we should handle and return"
            log_error_response_query(
                index=index,
                model="test",
                prompt=prompt_dict["prompt"],
                error_as_string=error_msg,
            )
            raise ValueError(error_msg)
        else:
            # wait 15 seconds to simulate a running task
            await asyncio.sleep(15)

        response_text = "This is a test response"
        log_success_response_query(
            index=index,
            model="test",
            prompt=prompt_dict["prompt"],
            response_text=response_text,
        )

        prompt_dict["response"] = response_text
        return prompt_dict
