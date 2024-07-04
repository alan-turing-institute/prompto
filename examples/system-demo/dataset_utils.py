import json


def load_prompt_dicts(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]
