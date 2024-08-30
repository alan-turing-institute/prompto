import asyncio
import random
import time
from typing import Any

from prompto.apis.base import AsyncAPI
from prompto.settings import Settings
from prompto.utils import log_error_response_query, log_success_response_query


class TestAPI(AsyncAPI):
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

    async def query(self, prompt_dict: dict, index: int | str) -> dict:
        # get the raise_error parameter from the prompt_dict
        # if not either "True" or "False", we error 1/5 times
        generation_config = prompt_dict.get("parameters", {})
        raise_error_option = generation_config.get("raise_error", "")
        raise_error_type = generation_config.get("raise_error_type", "")

        if raise_error_option == "True":
            raise_error = True
        elif raise_error_option == "False":
            raise_error = False
        else:
            raise_error = random.randint(1, 5) == 1

        if raise_error:
            error_msg = "This is a test error which we should handle and return"
            log_error_response_query(
                index=index,
                model="test",
                prompt=prompt_dict["prompt"],
                error_as_string=error_msg,
                id=prompt_dict.get("id", "NA"),
            )
            if raise_error_type == "Exception":
                raise Exception(error_msg)
            else:
                raise ValueError(error_msg)
        else:
            await asyncio.sleep(1)

        response_text = "This is a test response"
        log_success_response_query(
            index=index,
            model="test",
            prompt=prompt_dict["prompt"],
            response_text=response_text,
            id=prompt_dict.get("id", "NA"),
        )

        prompt_dict["response"] = response_text
        return prompt_dict
