import os

from vertexai.generative_models import GenerationConfig, Image, Part


def parse_multimedia_dict(multimedia_dict: dict, media_folder: str) -> Part:
    """
    Parse multimedia dictionary and create Vertex AI Part object.
    Expected keys:
    - type: str, multimedia type, one of ["image", "video", "text"]
    - media: str, file location (if type is image or video) or text (if type is text)
    - mime_type: str, mime type of the video file, only required for video type

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
        if type == "image":
            return Part.from_image(Image.load_from_file(media_file_path))
        elif type == "video":
            mime_type = multimedia_dict.get("mime_type")
            if mime_type is None:
                raise ValueError("Mime type is not specified. Required for video")
            return Part.from_data(
                open(media_file_path, "rb").read(), mime_type="video/mp4"
            )
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


def process_response(response: dict) -> str:
    response_text = response.candidates[0].content.parts[0].text
    return response_text


def process_safety_attributes(response: dict) -> dict:
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


def check_environment_variables() -> list[Exception]:
    # check the required environment variables are set
    issues = []
    required_env_vars = ["GEMINI_PROJECT_ID", "GEMINI_LOCATION"]
    for var in required_env_vars:
        if var not in os.environ:
            issues.append(ValueError(f"Environment variable {var} is not set"))

    other_env_vars = ["GEMINI_MODEL_ID"]
    for var in other_env_vars:
        if var not in os.environ:
            issues.append(Warning(f"Environment variable {var} is not set"))

    return issues


def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
    # check the parameter settings are valid
    issues = []

    # if safety_filter is provided, check that it's one of the valid options
    if "safety_filter" in prompt_dict and prompt_dict["safety_filter"] not in [
        "none",
        "few",
        "some",
        "default",
        "most",
    ]:
        issues.append(ValueError("Invalid safety_filter value"))

    # if generation_config is provided, check that it can create a valid GenerationConfig object
    if "parameters" in prompt_dict:
        try:
            GenerationConfig(**prompt_dict["parameters"])
        except TypeError as err:
            issues.append(TypeError(f"Invalid generation_config parameter: {err}"))
        except Exception as err:
            issues.append(ValueError(f"Invalid generation_config parameter: {err}"))

    return issues
