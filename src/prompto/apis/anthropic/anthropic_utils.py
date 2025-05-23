import base64
import os

from anthropic.types.message import Message

anthropic_chat_roles = set(["user", "assistant"])


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_content_value(content: dict | str, media_folder: str) -> dict:
    """
    Parse content dictionary and create a dictionary input for Anthropic API.
    If content is a string, a dictionary to represent a text object is returned.
    If content is a dictionary, expected keys are:
    - type: str, multimedia type, one of ["text", "image"]

    If type is "text", expect a key "text" with the text content.
    If type is "image", expect a key "source" which is a dictionary with keys:
    - url: str, URL of the image (can be a local path or a URL starting with "https://")
    - detail: str, optional detail parameter (default is "auto)

    Parameters
    ----------
    content : dict | str
        Either a dictionary or a string which defines a multimodal object.
    media_folder : str
        Folder where media files are stored ({data_folder}/media).

    Returns
    -------
    dict
        Dictionary which defines a text or image object
    """
    if isinstance(content, str):
        return {"type": "text", "text": content}

    # read multimedia type
    type = content.get("type")
    if type is None:
        raise ValueError("Multimedia type is not specified")

    # create dictionary based on multimedia type
    if type == "text":
        # read file location
        text = content.get("text")
        if text is None:
            raise ValueError(
                "Got type == 'text', but 'text' is not a key in the content dictionary"
            )

        return {"type": "text", "text": text}
    else:
        if type == "image":
            # read file location
            source = content.get("source")
            if source is None:
                raise ValueError(
                    "Got type == 'image', but 'source' is not a key in the content dictionary"
                )

            if not isinstance(source, dict):
                raise ValueError(
                    "Got type == 'image', but 'source' is not a dictionary"
                )

            # get media type
            media_type = source.get("media_type")
            if media_type is None:
                raise ValueError(
                    "Got type == 'image', but 'media_type' is not a key in the content['source'] dictionary"
                )

            # get image source
            media = source.get("media")
            if media is None:
                raise ValueError(
                    "Got type == 'image', but 'media' is not a key in the content['source'] dictionary"
                )

            # url is a local path and needs to be encoded to base64
            image_path = os.path.join(media_folder, media)
            base64_image = encode_image(image_path)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_image,
                },
            }
        else:
            raise ValueError(f"Unsupported multimedia type: {type}")


def parse_content(
    contents: list[dict | str] | dict | str, media_folder: str
) -> list[dict]:
    """
    Parse contents data and create a list of multimedia data objects.
    If contents is a single dictionary, a list with a single multimedia data object is returned.

    Parameters
    ----------
    contents : list[dict | str] | dict | str
        Contents data to parse and create Part object(s).
        Can be a list of dictionaries and strings, or a single dictionary or string.
    media_folder : str
        Folder where media files are stored ({data_folder}/media).

    Returns
    -------
    list[dict]
        List of dictionaries each defining a text or image object
    """
    # convert to list[dict | str]
    if isinstance(contents, dict) or isinstance(contents, str):
        contents = [contents]

    return [parse_content_value(p, media_folder=media_folder) for p in contents]


def convert_dict_to_input(content_dict: dict, media_folder: str) -> dict:
    """
    Convert dictionary to an input that can be used by the Anthropic API.
    The output is a dictionary with keys "role" and "contents".

    Parameters
    ----------
    content_dict : dict
        Content dictionary with keys "role" and "content" where
        the values are strings.
    media_folder : str
        Folder where media files are stored ({data_folder}/media).

    Returns
    -------
    dict
        dict with keys "role" and "contents"  where the value of
        role is either "user" or "model" and the value of
        contents is a list of inputs to make up an input (which can include
        text or image/video inputs).
    """
    if "role" not in content_dict:
        raise KeyError("role key is missing in content dictionary")
    if "content" not in content_dict:
        raise KeyError("content key is missing in content dictionary")

    return {
        "role": content_dict["role"],
        "content": parse_content(
            content_dict["content"],
            media_folder=media_folder,
        ),
    }


def process_response(response: Message) -> str | list[str]:
    """
    Helper function to process the response from the Anthropic API.

    Parameters
    ----------
    response : Message
        The response from the Anthropic API

    Returns
    -------
    str | list[str]
        The processed response. If there are multiple responses,
        a list of strings is returned, otherwise a single string is returned
    """
    assert isinstance(response, Message), f"Unsupported response type: {type(response)}"
    if len(response.content) == 0:
        return ""
    elif len(response.content) == 1:
        return response.content[0].text
    else:
        return [choice.text for choice in response.content]
