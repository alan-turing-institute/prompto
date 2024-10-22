import base64
import os

from openai.types.chat import ChatCompletion
from openai.types.completion import Completion

openai_chat_roles = set(["system", "user", "assistant"])


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_contents_value(content: dict | str, media_folder: str) -> dict:
    """
    Parse multimedia dictionary and create a dictionary input for OpenAI API.
    If content is a string, a dictionary to represent a text object is returned.
    If content is a dictionary, expected keys are:
    - type: str, multimedia type, one of ["text", "image_url"]

    If type is "text", expect a key "text" with the text content.
    If type is "image_url", expect a key "image_url" which is a dictionary with keys:
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
        Dictionary which defines a text or image_url object
    """
    if isinstance(content, str):
        return {"type": "text", "text": content}

    # read multimedia type
    type = content.get("type")
    if type is None:
        raise ValueError("Multimedia type is not specified")

    # create Part object based on multimedia type
    if type == "text":
        # read file location
        text = content.get("text")
        if text is None:
            raise ValueError(
                "Got type == 'text', but 'text' is not a key in the content dictionary"
            )

        return {"type": "text", "text": text}
    else:
        if type == "image_url":
            # read file location
            image_url = content.get("image_url")
            if image_url is None:
                raise ValueError(
                    "Got type == 'image_url', but 'image_url' is not a key in the content dictionary"
                )

            # get url (can be either a local path or a URL starting with "https://")
            url = image_url.get("url")
            if url is None:
                raise ValueError(
                    "Got type == 'image_url', but 'url' is not a key in the content['image_url'] dictionary"
                )

            # get detail parameter (default is "auto")
            detail = image_url.get("detail")
            if detail is None:
                detail = "auto"

            if url.startswith("https://"):
                return {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": detail},
                }

            # url is a local path and needs to be encoded to base64
            url_path = os.path.join(media_folder, url)
            base64_image = encode_image(url_path)
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": detail,
                },
            }
        else:
            raise ValueError(f"Unsupported multimedia type: {type}")


def parse_contents(
    contents: list[dict | str] | dict | str, media_folder: str
) -> list[dict]:
    """
    Parse multimedia data and create a list of multimedia data objects.
    If multimedia is a single dictionary, a list with a single multimedia data object is returned.

    Parameters
    ----------
    multimedia : list[dict | str] | dict | str
        Multimedia data to parse and create Part object(s).
        Can be a list of dictionaries and strings, or a single dictionary or string.
    media_folder : str
        Folder where media files are stored ({data_folder}/media).

    Returns
    -------
    list[dict]
        List of dictionaries each defining a text or image_url object
    """
    # convert to list[dict | str]
    if isinstance(contents, dict) or isinstance(contents, str):
        contents = [contents]

    return [parse_contents_value(p, media_folder=media_folder) for p in contents]


def convert_dict_to_input(content_dict: dict, media_folder: str) -> dict:
    """
    Convert dictionary to an input that can be used by the OpenAI API.
    The output is a dictionary with keys "role" and "contents".

    Parameters
    ----------
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
        raise KeyError("Role key is missing in content dictionary")
    if "contents" not in content_dict:
        raise KeyError("Parts key is missing in content dictionary")

    return {
        "role": content_dict["role"],
        "content": parse_contents(
            content_dict["content"],
            media_folder=media_folder,
        ),
    }


def process_response(response: ChatCompletion | Completion) -> str | list[str]:
    """
    Helper function to process the response from the OpenAI API.

    Parameters
    ----------
    response : ChatCompletion | Completion
        The response from the OpenAI API

    Returns
    -------
    str | list[str]
        The processed response. If there are multiple responses,
        a list of strings is returned, otherwise a single string is returned
    """
    if isinstance(response, ChatCompletion):
        if len(response.choices) == 0:
            return ""
        elif len(response.choices) == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]
    elif isinstance(response, Completion):
        if len(response.choices) == 0:
            return ""
        elif len(response.choices) == 1:
            return response.choices[0].text
        else:
            return [choice.text for choice in response.choices]
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")
