import logging
from typing import Callable


def obtain_scoring_functions(
    scorer: str | list[str], scoring_functions_dict: dict[str, Callable]
) -> list[Callable]:
    """
    Check if the scorer(s) provided are in the scoring_functions_dict.

    Parameters
    ----------
    scorer : str | list[str]
        A single scorer or a list of scorers to check if they
        are keys in the scoring_functions_dict dictionary
    scoring_functions_dict : dict[str, Callable]
        A dictionary of scoring functions with the keys as the
        scorer names and the values as the scoring functions

    Returns
    -------
    list[Callable]
        List of scoring functions that correspond to the scorers
    """
    if isinstance(scorer, str):
        scorer = [scorer]

    functions = []
    for s in scorer:
        if not isinstance(s, str):
            raise TypeError("If scorer is a list, each element must be a string")
        if s not in scoring_functions_dict.keys():
            raise KeyError(
                f"Scorer '{s}' is not a key in scoring_functions_dict. "
                f"Available scorers are: {list(scoring_functions_dict.keys())}"
            )

        functions.append(scoring_functions_dict[s])

    logging.info(f"Scoring functions to be used: {scorer}")
    return functions


def match(prompt_dict: dict) -> dict:
    """
    Returns a True if the prompt_dict["response"]
    is equal to the prompt_dict["expected_response"].

    Parameters
    ----------
    prompt_dict : dict
        A dictionary containing a "response" and
        "expected_response" key

    Returns
    -------
    dict
        A dictionary containing the "match" key with
        a boolean value of the comparison between the
        "response" and "expected_response" keys.
    """
    prompt_dict["match"] = prompt_dict["response"] == prompt_dict["expected_response"]
    return prompt_dict


def includes(prompt_dict: dict) -> dict:
    """
    Returns a True if the prompt_dict["response"]
    includes the prompt_dict["expected_response"].

    Parameters
    ----------
    prompt_dict : dict
        A dictionary containing a "response" and
        "expected_response" key

    Returns
    -------
    dict
        A dictionary containing the "includes" key with
        a boolean value of the comparison between the
        "response" and "expected_response" keys.
    """
    prompt_dict["includes"] = (
        prompt_dict["expected_response"] in prompt_dict["response"]
    )
    return prompt_dict


SCORING_FUNCTIONS = {
    "match": match,
    "includes": includes,
}
