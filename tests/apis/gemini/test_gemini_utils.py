import pytest
from google.genai.types import Content, Part

from prompto.apis.gemini.gemini_utils import convert_history_dict_to_content

from .test_gemini import prompt_dict_history


def test_convert_history_dict_to_content(prompt_dict_history):

    media_folder = "media_folder"
    mock_client = None

    expected_result_list = [
        Content(parts=[Part(text="test system prompt")], role="system"),
        Content(parts=[Part(text="user message")], role="user"),
    ]

    # There is no uploaded media in the prompt_dict_chat, hence the
    # client is not required, and we can pass `None`.
    prompt_list = prompt_dict_history["prompt"]

    for content_dict, expected_result in zip(prompt_list, expected_result_list):
        actual_result = convert_history_dict_to_content(
            content_dict, media_folder, mock_client
        )
        assert (
            actual_result == expected_result
        ), f"Expected {expected_result}, but got {actual_result}"
