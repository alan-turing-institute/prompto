import json
from typing import Any


def generate_experiment_1_file(
    path: str, prompts: list[str], api: str, model_name: str, params: dict[str, Any]
) -> None:
    # function to generate an experiment file with prompts for a
    # single API, model_name, and parameters configuration
    index = 0
    with open(path, "w") as f:
        for prompt in prompts:
            f.write(
                json.dumps(
                    {
                        "id": index,
                        "api": api,
                        "model_name": model_name,
                        "prompt": prompt,
                        "parameters": params,
                    }
                )
            )
            f.write("\n")
            index += 1


def generate_experiment_2_file(
    path: str,
    prompts: list[str],
    api: list[str],
    model_name: list[str],
    params: list[dict[str, Any]],
) -> None:
    # function to generate an experiment file with prompts for multiple
    # API, model_name, and parameters configurations
    if len(api) != len(model_name) or len(model_name) != len(params):
        raise ValueError("Length mismatch between api, model_name, and params lists")

    index = 0
    with open(path, "w") as f:
        for i in range(len(api)):
            for prompt in prompts:
                f.write(
                    json.dumps(
                        {
                            "id": index,
                            "api": api[i],
                            "model_name": model_name[i],
                            "prompt": prompt,
                            "parameters": params[i],
                        }
                    )
                )
                f.write("\n")
                index += 1


def generate_experiment_3_file(
    path: str,
    prompts: list[str],
    api: str,
    model_name: list[str],
    params: dict[str, Any],
) -> None:
    # function to generate an experiment file with prompts for a
    # single API and parameters configuration but for multiple model_names

    index = 0
    with open(path, "w") as f:
        for i in range(len(model_name)):
            for prompt in prompts:
                f.write(
                    json.dumps(
                        {
                            "id": index,
                            "api": api,
                            "model_name": model_name[i],
                            "prompt": prompt,
                            "parameters": params,
                        }
                    )
                )
                f.write("\n")
                index += 1


def load_prompts(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)


def load_prompt_dicts(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]
