import os
from enum import Enum

from openai.types.chat import ChatCompletion
from openai.types.completion import Completion


class ChatRoles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def process_response(response: ChatCompletion | Completion) -> str:
    if isinstance(response, ChatCompletion):
        return response.choices[0].message.content
    elif isinstance(response, Completion):
        return response.choices[0].text
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")


def check_environment_variables() -> list[Exception]:
    # check the required environment variables are set
    issues = []
    required_env_vars = ["OPENAI_API_KEY"]
    for var in required_env_vars:
        if var not in os.environ:
            issues.append(ValueError(f"Environment variable {var} is not set"))

    other_env_vars = ["OPENAI_MODEL_NAME"]
    for var in other_env_vars:
        if var not in os.environ:
            issues.append(Warning(f"Environment variable {var} is not set"))

    return issues


def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
    issues = []

    # check prompt is of the right type
    match prompt_dict["prompt"]:
        case str(_):
            pass
        case [str(_)]:
            pass
        case [{"role": role, "content": _}, *rest]:
            if role in ChatRoles and all(
                [
                    set(d.keys()) == {"role", "content"} and d["role"] in ChatRoles
                    for d in rest
                ]
            ):
                pass
        case _:
            issues.append(
                TypeError(
                    "If model == 'openai', then the prompt must be a str, list[str], or "
                    "list[dict[str,str]] where the dictionary contains the keys 'role' and "
                    "'content' only, and the values for 'role' must be one of 'system', 'user' or "
                    "'assistant'"
                )
            )

    # check if the model_name is set as an environment variable if not provided in the prompt_dict
    if "model_name" not in prompt_dict:
        req_var = "OPENAI_MODEL_NAME"
        if req_var not in os.environ:
            issues.append(
                ValueError(
                    f"model_name is not set and environment variable {req_var} is not set"
                )
            )

    # if mode is passed, check it is a valid value
    if "mode" in prompt_dict and prompt_dict["mode"] not in ["chat", "completion"]:
        issues.append(
            ValueError(
                f"Invalid mode value. Must be 'chat' or 'completion', not {prompt_dict['mode']}"
            )
        )

    # TODO: add checks for prompt_dict["parameters"] being
    # valid arguments for OpenAI API without hardcoding

    return issues
