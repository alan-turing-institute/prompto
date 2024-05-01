import os

import openai
from openai import AsyncOpenAI, OpenAI
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
    # only check for the optional environment variables to define the "default" model
    # which is used if the model name is not provided in the prompt dictionary
    issues = []

    # check the optional environment variables are set and warn if not
    other_env_vars = ["HUGGINGFACE_TGI_API_KEY", "HUGGINGFACE_TGI_API_ENDPOINT"]
    for var in other_env_vars:
        if var not in os.environ:
            issues.append(Warning(f"Environment variable {var} is not set"))

    return issues


def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
    # for Huggingface TGI, there's specific environment variables that need to be set
    # for different model_name values
    issues = []

    if "model_name" not in prompt_dict:
        # use the default API endpoint and endpoint that must be provided as environment variables
        # check the required environment variables are set
        required_env_vars = ["HUGGINGFACE_TGI_API_ENDPOINT"]
        for var in required_env_vars:
            if var not in os.environ:
                issues.append(ValueError(f"Environment variable {var} is not set"))

        # check the optional environment variables are set and warn if not
        other_env_vars = ["HUGGINGFACE_TGI_API_KEY"]
        for var in other_env_vars:
            if var not in os.environ:
                issues.append(Warning(f"Environment variable {var} is not set"))
    else:
        model_name = prompt_dict["model_name"]
        # check the required environment variables are set
        required_env_vars = [
            f"HUGGINGFACE_TGI_API_ENDPOINT_{model_name}",
        ]
        for var in required_env_vars:
            if var not in os.environ:
                issues.append(ValueError(f"Environment variable {var} is not set"))

        # check the optional environment variables are set and warn if not
        other_env_vars = [
            f"HUGGINGFACE_TGI_API_KEY_{model_name}",
        ]
        for var in other_env_vars:
            if var not in os.environ:
                issues.append(Warning(f"Environment variable {var} is not set"))

    return issues


def obtain_model_inputs(
    prompt_dict: dict, async_client: bool
) -> tuple[str, str, dict, OpenAI | AsyncOpenAI, str]:
    # obtain the prompt from the prompt dictionary
    prompt = prompt_dict["prompt"]

    # obtain model name
    model_name = prompt_dict.get("model_name", None)
    if model_name is None:
        api_key_env_var = "HUGGINGFACE_TGI_API_KEY"
        api_endpoint_env_var = "HUGGINGFACE_TGI_API_ENDPOINT"
    else:
        api_key_env_var = f"HUGGINGFACE_TGI_API_KEY_{model_name}"
        api_endpoint_env_var = f"HUGGINGFACE_TGI_API_ENDPOINT_{model_name}"

    API_KEY = os.environ.get(api_key_env_var)
    API_ENDPOINT = os.environ.get(api_endpoint_env_var)

    if API_KEY is None:
        # need pass string to initialise OpenAI client
        API_KEY = "-"

    if API_ENDPOINT is None:
        raise ValueError(f"{api_endpoint_env_var} environment variable not found")

    openai.api_key = API_KEY
    openai.api_type = API_ENDPOINT

    if async_client:
        client = AsyncOpenAI(
            base_url=f"{API_ENDPOINT}/v1/",
            api_key=API_KEY,
        )
    else:
        client = OpenAI(
            base_url=f"{API_ENDPOINT}/v1/",
            api_key=API_KEY,
        )

    # get parameters dict (if any)
    generation_config = prompt_dict.get("parameters", None)
    if generation_config is None:
        generation_config = {}
    if type(generation_config) is not dict:
        raise TypeError(
            f"parameters must be a dictionary, not {type(generation_config)}"
        )

    # add in default parameters
    default_generation_config = {
        "max_tokens": 2048,
        "temperature": 0.7,
        "n": 1,
    }
    for key, value in default_generation_config.items():
        if key not in generation_config:
            generation_config[key] = value

    # obtain mode (default is chat)
    mode = prompt_dict.get("mode", "query")
    if mode not in ["query", "chat"]:
        raise ValueError(f"mode must be 'query' or 'chat', not {mode}")

    return prompt, model_name, generation_config, client, mode
