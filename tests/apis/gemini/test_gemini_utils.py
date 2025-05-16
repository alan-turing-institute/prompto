import pytest
from google.genai.types import Candidate, Content, FinishReason, Part

from prompto.apis.gemini.gemini_utils import (
    convert_history_dict_to_content,
    process_response,
)

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


def test_process_response():
    """
    Test the process_response function.
    """
    # Some example responses, with and without thinking included.
    non_thinking_candidates = [
        Candidate(
            content=Content(
                parts=[
                    Part(
                        video_metadata=None,
                        thought=None,
                        inline_data=None,
                        code_execution_result=None,
                        executable_code=None,
                        file_data=None,
                        function_call=None,
                        function_response=None,
                        text="A spontaneous answer...",
                    ),
                ],
                role="model",
            ),
            citation_metadata=None,
            finish_message=None,
            token_count=None,
            finish_reason=FinishReason.STOP,
            avg_logprobs=None,
            grounding_metadata=None,
            index=0,
            logprobs_result=None,
            safety_ratings=None,
        )
    ]

    thinking_candidates = [
        Candidate(
            content=Content(
                parts=[
                    Part(
                        video_metadata=None,
                        thought=True,
                        inline_data=None,
                        code_execution_result=None,
                        executable_code=None,
                        file_data=None,
                        function_call=None,
                        function_response=None,
                        text="Some thinking...",
                    ),
                    Part(
                        video_metadata=None,
                        thought=None,
                        inline_data=None,
                        code_execution_result=None,
                        executable_code=None,
                        file_data=None,
                        function_call=None,
                        function_response=None,
                        text="A thought out answer...",
                    ),
                ],
                role="model",
            ),
            citation_metadata=None,
            finish_message=None,
            token_count=None,
            finish_reason=FinishReason.STOP,
            avg_logprobs=None,
            grounding_metadata=None,
            index=0,
            logprobs_result=None,
            safety_ratings=None,
        )
    ]

    # Each test case is a tuple of (candidates, expected_answer, expected_thoughts)
    test_cases = [
        (non_thinking_candidates, "A spontaneous answer...", [None]),
        (thinking_candidates, "A thought out answer...", ["Some thinking..."]),
    ]

    for candidates, expected_answer, expected_thoughts in test_cases:

        # Create a mock response - the candidates key is the only one used in the
        # process_response function.
        response = {
            "candidates": candidates,
        }

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
