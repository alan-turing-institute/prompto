import os

from openai.types.chat import ChatCompletion
from openai.types.completion import Completion


def process_response(response: ChatCompletion | Completion) -> str:
    if isinstance(response, ChatCompletion):
        return response.choices[0].message.content
    elif isinstance(response, Completion):
        return response.choices[0].text
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")


def check_environment_variables() -> list[Exception]:
    issues = []

    # check the required environment variables are set
    required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_ENDPOINT"]
    for var in required_env_vars:
        if var not in os.environ:
            issues.append(ValueError(f"Environment variable {var} is not set"))

    # check the optional environment variables are set and warn if not
    other_env_vars = ["AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_MODEL_NAME"]
    for var in other_env_vars:
        if var not in os.environ:
            issues.append(Warning(f"Environment variable {var} is not set"))

    return issues


def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
    # TODO: add checks for prompt_dict["parameters"]["generation_config"] being
    # valid arguments for OpenAI API without hardcoding
    return []
