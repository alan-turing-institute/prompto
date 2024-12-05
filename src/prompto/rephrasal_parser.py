import logging
from typing import Callable

import regex as re


def obtain_parser_functions(
    parser: str | list[str], parser_functions_dict: dict[str, Callable]
) -> list[Callable]:
    """
    Check if the parser(s) provided are in the parser_functions_dict.

    Parameters
    ----------
    parser : str | list[str]
        A single parser or a list of parsers to check if they
        are keys in the parser_functions_dict dictionary
    parser_functions_dict : dict[str, Callable]
        A dictionary of parser functions with the keys as the
        parser names and the values as the parser functions

    Returns
    -------
    list[Callable]
        List of parser functions that correspond to the parsers
    """
    if isinstance(parser, str):
        parser = [parser]

    functions = []
    for p in parser:
        if not isinstance(p, str):
            raise TypeError("If parser is a list, each element must be a string")
        if p not in parser_functions_dict.keys():
            raise KeyError(
                f"Parser '{p}' is not a key in parser_functions_dict. "
                f"Available parsers are: {list(parser_functions_dict.keys())}"
            )

        functions.append(parser_functions_dict[p])

    logging.info(f"parser functions to be used: {parser}")
    return functions


def remove_brackets(text: str) -> str:
    # regex to remove anything brackets and anything between them
    return re.sub(r"\(.*?\)", "", text)


def remove_quotation_marks(text: str) -> str:
    # remove quotation marks only if they are at the beginning and end of the string
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def split_numbered_list(text: str) -> list[str]:
    # regex pattern matches:
    # - Starts with one or more digits (\d+) at the
    #   beginning of the line (^) or after a newline
    # - followed by a period (\.)
    # - followed by optional whitespace (\s*)
    pattern = r"(?<=\n|^)\d+\.\s*"

    # split the text and clean each part
    parts = re.split(pattern, text)

    # strip whitespace and newlines from each part
    parts = [remove_quotation_marks(remove_brackets(part).strip()) for part in parts]

    # remove empty strings from the beginning if they exist
    if parts and not parts[0]:
        parts = parts[1:]

    return parts


PARSER_FUNCTIONS = {
    "split_numbered_list": split_numbered_list,
}
