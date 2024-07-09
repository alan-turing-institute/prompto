import os

from vertexai.generative_models import Content, Image, Part


def parse_multimedia_dict(multimedia_dict: dict, media_folder: str) -> Part:
    """
    Parse multimedia dictionary and create Vertex AI Part object.
    Expected keys:
    - type: str, multimedia type, one of ["image", "video", "uri" "text"]
    - media: str, file location (if type is image or video) or text (if type is text)
    - mime_type: str, mime type of the video file, only required for video and uri types

    Parameters
    ----------
    multimedia_dict : dict
        Dictionary with multimedia data to parse and create Part object

    Returns
    -------
    Part
        Vertex AI Part object created from multimedia data
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
        return Part.from_text(media)
    else:
        media_file_path = os.path.join(media_folder, media)
        mime_type = multimedia_dict.get("mime_type")
        if type == "image":
            return Part.from_image(Image.load_from_file(media_file_path))
        elif type == "video":
            if mime_type is None:
                raise ValueError("Mime type is not specified. Required for video")
            return Part.from_data(
                open(media_file_path, "rb").read(), mime_type="video/mp4"
            )
        elif type == "uri":
            if mime_type is None:
                raise ValueError("Mime type is not specified. Required for uri")
            return Part.from_uri(uri=media, mime_type=mime_type)
        else:
            raise ValueError(f"Unsupported multimedia type: {type}")


def parse_multimedia(multimedia: list[dict] | dict, media_folder: str) -> list[Part]:
    """
    Parse multimedia data and create a list of Vertex AI Part object(s).
    If multimedia is a single dictionary, a list with a single Part object is returned.

    Parameters
    ----------
    multimedia : list[dict] | dict
        Multimedia data to parse and create Part object(s).
        Can be a list of multimedia dictionaries or a single multimedia dictionary.

    Returns
    -------
    list[Part] | Part
        List of Vertex AI Part object(s) created from multimedia data
    """
    if isinstance(multimedia, dict):
        return [parse_multimedia_dict(multimedia, media_folder=media_folder)]
    else:
        return [parse_multimedia_dict(m, media_folder=media_folder) for m in multimedia]


def dict_to_content(content_dict: dict):
    """
    Convert content dictionary to Vertex AI Content object.

    Parameters
    ----------
    content_dict : dict
        Content dictionary with keys "role" and "parts" where
        the values are strings.

    Returns
    -------
    Content
        Vertex AI Content object created from content dictionary
    """
    if "role" not in content_dict:
        raise KeyError("Role key is missing in content dictionary")
    if "parts" not in content_dict:
        raise KeyError("Parts key is missing in content dictionary")

    return Content(
        role=content_dict["role"], parts=[Part.from_text(content_dict["parts"])]
    )
