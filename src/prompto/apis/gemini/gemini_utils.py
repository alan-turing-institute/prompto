import os

import PIL.Image
from google.generativeai import get_file

gemini_chat_roles = set(["user", "model"])


def parse_parts_value(part: dict | str, media_folder: str) -> any:
    """
    Parse multimedia dictionary and create Vertex AI Part object.
    Expected keys:
    - type: str, multimedia type, one of ["text", "image", "file"]
    - media: str, file location (if type is image or file), text (if type is text)

    Parameters
    ----------
    part : dict | str
        Either a dictionary or a string which defines a multimodal object.

    Returns
    -------
    any
        Multimedia data object
    """
    if isinstance(part, str):
        return part

    # read multimedia type
    type = part.get("type")
    if type is None:
        raise ValueError("Multimedia type is not specified")
    # read file location
    media = part.get("media")
    if media is None:
        raise ValueError("File location is not specified")

    # create Part object based on multimedia type
    if type == "text":
        return media
    else:
        if type == "image":
            media_file_path = os.path.join(media_folder, media)
            return PIL.Image.open(media_file_path)
        elif type == "file":
            try:
                return get_file(name=media)
            except Exception as err:
                raise ValueError(
                    f"Failed to get file: {media} due to error: {type(err).__name__} - {err}"
                )
        else:
            raise ValueError(f"Unsupported multimedia type: {type}")


def parse_parts(parts: list[dict | str] | dict | str, media_folder: str) -> list[any]:
    """
    Parse multimedia data and create a list of multimedia data objects.
    If multimedia is a single dictionary, a list with a single multimedia data object is returned.

    Parameters
    ----------
    multimedia : list[dict] | dict
        Multimedia data to parse and create Part object(s).
        Can be a list of multimedia dictionaries or a single multimedia dictionary.

    Returns
    -------
    list[any] | any
        List of multimedia data object(s) created from the input multimedia data
    """
    # convert to list[dict | str]
    if isinstance(parts, dict) or isinstance(parts, str):
        parts = [parts]

    return [parse_parts_value(p, media_folder=media_folder) for p in parts]


def convert_dict_to_input(content_dict: dict, media_folder: str) -> dict:
    """
    Convert dictionary to an input that can be used by the Gemini API.
    The output is a dictionary with keys "role" and "parts".

    Parameters
    ----------
    content_dict : dict
        Content dictionary with keys "role" and "parts"strings.

    Returns
    -------
    dict
        dict with keys "role" and "parts"  where the value of
        role is either "user" or "model" and the value of
        parts is a list of inputs to make up an input (which can include
        text or image/video inputs).
    """
    if "role" not in content_dict:
        raise KeyError("Role key is missing in content dictionary")
    if "parts" not in content_dict:
        raise KeyError("Parts key is missing in content dictionary")

    return {
        "role": content_dict["role"],
        "parts": parse_parts(
            content_dict["parts"],
            media_folder=media_folder,
        ),
    }


def process_response(response: dict) -> str:
    """
    Helper function to process the response from Gemini API.

    Parameters
    ----------
    response : dict
        The response from the Gemini API as a dictionary

    Returns
    -------
    str
        The processed response text as a string
    """
    response_text = response.candidates[0].content.parts[0].text
    return response_text


def process_safety_attributes(response: dict) -> dict:
    """
    Helper function to process the safety attributes from Gemini API.

    Parameters
    ----------
    response : dict
        The response from the Gemini API as a dictionary

    Returns
    -------
    dict
        The safety attributes as a dictionary with category names as keys
        and their respective probabilities as values. Additionally,
        the dictionary contains a key 'blocked' with a list of booleans
        indicating whether each category is blocked, and 'finish_reason'
    """
    safety_attributes = {
        x.category.name: str(x.probability)
        for x in response.candidates[0].safety_ratings
    }
    # list of booleans indicating whether each category is blocked
    safety_attributes["blocked"] = str(
        [x.blocked for x in response.candidates[0].safety_ratings]
    )
    safety_attributes["finish_reason"] = str(response.candidates[0].finish_reason.name)

    return safety_attributes
