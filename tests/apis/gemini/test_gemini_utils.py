import pytest
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    GenerateContentResponse,
    Part,
)

from prompto.apis.gemini.gemini_utils import (
    convert_history_dict_to_content,
    process_response,
    process_thoughts,
)

from .test_gemini import non_thinking_response, prompt_dict_history, thinking_response


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


def test_process_response(thinking_response, non_thinking_response):
    """
    Test the process_response function.
    """
    # Each test case is a tuple of (response, expected_answer, expected_thoughts)
    test_cases = [
        (non_thinking_response, "A spontaneous answer", []),
        (thinking_response, "A thought out answer", ["Some thinking"]),
    ]

    for response, expected_answer, expected_thoughts in test_cases:
        # Call the function with the test case
        actual_answer = process_response(response)
        actual_thoughts = process_thoughts(response)

        # Assert the expected response
        assert (
            actual_answer == expected_answer
        ), f"Expected {expected_answer}, but got {actual_answer}"

        assert isinstance(
            actual_thoughts, list
        ), f"Expected a list, but got {type(actual_thoughts)}"
        assert (
            actual_thoughts == expected_thoughts
        ), f"Expected {expected_thoughts}, but got {actual_thoughts}"
