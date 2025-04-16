import os

from vertexai.generative_models import Content, Image, Part


def parse_parts_value(part: dict | str, media_folder: str) -> Part:
    """
    Create Vertex AI Part objects from a dictionary or string.
    If parts is a string, a Part object with text is created.
    If parts is a dictionary, expected keys are:
    - type: str, multimedia type, one of ["image", "video", "uri" "text"]
    - media: str, file location (if type is image or video) or text (if type is text).
      This can be either a local file path (relative to the media folder) or a GCS URI.
    - mime_type: str, mime type of the image orvideo file, only required
      if using a GCS URI, or using a local video file

    Parameters
    ----------
    part : dict | str
        Either a dictionary or a string which defines a Part object.
    media_folder : str
        Folder where media files are stored ({data_folder}/media).

    Returns
    -------
    Part
        Vertex AI Part object created from multimedia data
    """
    if isinstance(part, str):
        return Part.from_text(part)

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
        return Part.from_text(media)
    else:
        media_file_path = os.path.join(media_folder, media)
        mime_type = part.get("mime_type")
        if type == "image":
            if media.startswith("gs://"):
                if mime_type is None:
                    raise ValueError(
                        "Mime type is not specified. Required for image type if media is a GCS URI"
                    )

                return Part.from_uri(uri=media, mime_type=mime_type)

            return Part.from_image(Image.load_from_file(media_file_path))
        elif type == "video":
            if mime_type is None:
                raise ValueError("Mime type is not specified. Required for video type")

            if media.startswith("gs://"):
                return Part.from_uri(uri=media, mime_type=mime_type)

            return Part.from_data(
                open(media_file_path, "rb").read(), mime_type="video/mp4"
            )
        elif type == "uri":
            if mime_type is None:
                raise ValueError("Mime type is not specified. Required for uri")
            return Part.from_uri(uri=media, mime_type=mime_type)
        else:
            raise ValueError(f"Unsupported multimedia type: {type}")


def parse_parts(parts: list[dict | str] | dict | str, media_folder: str) -> list[Part]:
    """
    Parse "parts" value and create a list of Vertex AI Part object(s).
    If parts is a single dictionary or a string, a list with a single Part
    object is returned, otherwise, a list of multiple Part objects is created.

    Parameters
    ----------
    parts : list[dict | str] | dict | str
        Corresponding to the "parts" value in the prompt.
        Can be a list of dictionaries and strings, or a single dictionary or string.
    media_folder : str
        Folder where media files are stored ({data_folder}/media).

    Returns
    -------
    list[Part]
        List of Vertex AI Part object(s) created from "parts" value in a prompt
    """
    # convert to list[dict | str]
    if isinstance(parts, dict) or isinstance(parts, str):
        parts = [parts]

    return [parse_parts_value(p, media_folder=media_folder) for p in parts]


def convert_dict_to_input(content_dict: dict, media_folder: str) -> Content:
    """
    Convert content dictionary to Vertex AI Content object.

    Parameters
    ----------
    content_dict : dict
        Content dictionary with keys "role" and "parts" where
        the values are strings.
    media_folder : str
        Folder where media files are stored ({data_folder}/media).

    Returns
    -------
    Content
        Vertex AI Content object created from content dictionary
    """
    if "role" not in content_dict:
        raise KeyError("role key is missing in content dictionary")
    if "parts" not in content_dict:
        raise KeyError("parts key is missing in content dictionary")

    return Content(
        role=content_dict["role"],
        parts=parse_parts(
            content_dict["parts"],
            media_folder=media_folder,
        ),
    )
