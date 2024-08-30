def obtain_scoring_functions(
    scorer: str | list[str], scoring_functions_dict: dict[str, callable]
) -> list[callable]:
    """
    Check if the scorer(s) provided are in the scoring_functions_dict.

    Parameters
    ----------
    scorer : str | list[str]
        A single scorer or a list of scorers to check if they
        are keys in the scoring_functions_dict dictionary
    scoring_functions_dict : dict[str, callable]
        A dictionary of scoring functions with the keys as the
        scorer names and the values as the scoring functions

    Returns
    -------
    list[callable]
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

    return functions


def match(prompt_dict: dict):
    """
    Returns a True if the prompt_dict["response"]
    is equal to the prompt_dict["expected_response"].

    Parameters
    ----------
    prompt_dict : dict
        A dictionary containing a "response" and
        "expected_response" key.
    """
    return prompt_dict["response"] == prompt_dict["expected_response"]


def includes(prompt_dict: dict):
    """
    Returns a True if the prompt_dict["response"]
    includes the prompt_dict["expected_response"].

    Parameters
    ----------
    prompt_dict : dict
        A dictionary containing a "response" and
        "expected_response" key.
    """
    return prompt_dict["expected_response"] in prompt_dict["response"]


SCORING_FUNCTIONS = {
    "includes": includes,
    "match": match,
}
