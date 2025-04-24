import os
from unittest.mock import patch

import pytest
from google.genai import Client
from google.genai.types import Part
from PIL import Image

from prompto.apis.gemini.gemini_utils import parse_parts_value


def test_parse_parts_value_text():
    part = "text"
    media_folder = "media"
    # There is no uploaded media in the prompt_dict_chat, hence the
    # client is not required, and we can pass `None`.
    mock_client = None
    actual_result = parse_parts_value(part, media_folder, mock_client)
    expected_result = Part(text="text")
    assert actual_result == expected_result


def test_parse_parts_value_image():
    # This is a string, which happens to use a keyword "image",
    # but it is not a key within dictionary.
    # This test simply asserts that the string is handled correctly
    # as a string.
    part = "image"
    media_folder = "media"
    # There is no uploaded media in the prompt_dict_chat, hence the
    # client is not required, and we can pass `None`.
    mock_client = None
    actual_result = parse_parts_value(part, media_folder, mock_client)
    expected_result = Part(text="image")
    assert actual_result == expected_result


def test_parse_parts_value_image_dict(tmp_path):

    # The part dictionary is an extract of a prompt_dict in an experiment
    # file. In this case there is a local image file but not a remote one.
    # The image should be loaded from the local file.
    part = {"type": "image", "media": "pantani_giro.jpg"}
    media_folder = tmp_path / "media"
    mock_client = None

    media_folder.mkdir(parents=True, exist_ok=True)

    # Create a mock image
    image_path = media_folder / "pantani_giro.jpg"
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_path)

    actual_result = parse_parts_value(part, str(media_folder), mock_client)

    # Assert the result
    assert actual_result.mode == "RGB"
    assert actual_result.size == (100, 100)
    assert actual_result.filename.endswith("pantani_giro.jpg")


def test_parse_parts_value_video_not_uploaded():
    part = {"type": "video", "media": "pantani_giro.mp4"}
    media_folder = "media"
    mock_client = None

    # Because the video is not uploaded, we expect a ValueError
    with pytest.raises(ValueError) as excinfo:
        parse_parts_value(part, media_folder, mock_client)

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

    # Replace the original `get` function with the mock
    with monkeypatch.context() as m:
        # Mock the get function
        client = Client(api_key="DUMMY")
        m.setattr(client.aio.files, "get", mock_get_file_no_op)
        # Assert that the mock function was called with the expected argument
        assert (
            client.aio.files.get(name="check mocked function")
            == "check mocked function"
        )

        expected_result = "file/123456"
        actual_result = parse_parts_value(part, media_folder, client)
        assert actual_result == expected_result
