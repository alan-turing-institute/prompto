import os

import PIL.Image
from google.generativeai import get_file

gemini_chat_roles = ["user", "model"]


def parse_multimedia_dict(multimedia_dict: dict, media_folder: str) -> any:
    """
    Parse multimedia dictionary and create Vertex AI Part object.
    Expected keys:
    - type: str, multimedia type, one of ["text", "image", "file"]
    - media: str, file location (if type is image or file), text (if type is text)

    Parameters
    ----------
    multimedia_dict : dict
        Dictionary with multimedia data to parse and create Part object

    Returns
    -------
    any
        Multimedia data object
    """
    # read multimedia type
    type = multimedia_dict.get("type")
    if type is None:
        raise ValueError("Multimedia type is not specified")
    # read file location
    media = multimedia_dict.get("media")
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


def parse_multimedia(multimedia: list[dict] | dict, media_folder: str) -> list[any]:
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
    if isinstance(multimedia, dict):
        return [parse_multimedia_dict(multimedia, media_folder=media_folder)]
    else:
        return [parse_multimedia_dict(m, media_folder=media_folder) for m in multimedia]


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
