import os
from unittest.mock import patch

import pytest
from PIL import Image

from prompto.apis.gemini.gemini_utils import parse_parts_value


def test_parse_parts_value_text():
    part = "text"
    media_folder = "media"
    result = parse_parts_value(part, media_folder)
    assert result == part


def test_parse_parts_value_image():
    part = "image"
    media_folder = "media"
    result = parse_parts_value(part, media_folder)
    assert result == part


def test_parse_parts_value_image_dict(tmp_path):

    # The part dictionary is an extract of a prompt_dict in an experiment
    # file. In this case there is a local image file but not a remote one.
    # The image should be loaded from the local file.
    part = {"type": "image", "media": "pantani_giro.jpg"}
    media_folder = tmp_path / "media"

    media_folder.mkdir(parents=True, exist_ok=True)

    # Create a mock image
    image_path = media_folder / "pantani_giro.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_path)

    actual_result = parse_parts_value(part, str(media_folder))

    # Assert the result
    assert actual_result.mode == "RGB"
    assert actual_result.size == (100, 100)
    assert actual_result.filename.endswith("pantani_giro.jpg")


def test_parse_parts_value_video_not_uploaded():
    part = {"type": "video", "media": "pantani_giro.mp4"}
    media_folder = "media"

    # Because the video is not uploaded, we expect a ValueError
    with pytest.raises(ValueError) as excinfo:
        parse_parts_value(part, media_folder)

    print(excinfo)
    assert "not uploaded" in str(excinfo.value)


# @patch("google.genai.client.files.get")
def test_parse_parts_value_video_uploaded(monkeypatch):
    part = {
        "type": "video",
        "media": "pantani_giro.mp4",
        "uploaded_filename": "file/123456",
    }
    # This directory is not used in the test but it is a required
    # parameter for the parse_parts_value function
    media_folder = "media"

    # Mock the `google.genai.Client().files.get`` function
    # The real `get` function returns the binary contents of the file
    # which would be tricky to assert.
    # Instead, we will just return the uploaded_filename
    mock_get_file_no_op = lambda name: name

    # Replace the original get_file function with the mock
    # ***It is important that the import statement used here is exactly the same as
    # the one in the gemini_utils.py file***
    import google.genai as genai

    with monkeypatch.context() as m:
        # Mock the get_file function
        client = genai.Client(api_key="DUMMY")
        m.setattr(client.files, "get", mock_get_file_no_op)
        # Assert that the mock function was called with the expected argument
        assert client.files.get(name="check mocked function") == "check mocked function"

        expected_result = "file/123456"
        actual_result = parse_parts_value(part, media_folder)
        assert actual_result == expected_result
