import logging
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from google import genai
from google.genai.types import FileState

from prompto.apis.gemini.gemini import GeminiAPI
from prompto.apis.gemini.gemini_media import (
    delete_uploaded_files,
    list_uploaded_files,
    remote_file_hash_base64,
    upload_media_files,
    wait_for_processing,
)
from prompto.upload_media import _create_settings


def test_remote_file_hash_base64():

    # The example hashes are for the strings "hash1", "hash2", "hash3" eg:
    # >>> "hash1".encode("utf-8").hex()
    # '6861736831'
    test_cases = [
        (
            Mock(dummy_name="file1", sha256_hash="6861736831"),
            "aGFzaDE=",
        ),
        (
            Mock(dummy_name="file2", sha256_hash="6861736832"),
            "aGFzaDI=",
        ),
        (
            Mock(dummy_name="file3", sha256_hash="6861736833"),
            "aGFzaDM=",
        ),
    ]

    # We need to juggle with the mock names, because we can't set them
    # directly in the constructor. See these docs for more details:
    # https://docs.python.org/3/library/unittest.mock.html#mock-names-and-the-name-attribute
    for mock_file, expected_hash in test_cases:
        mock_file.configure_mock(name=mock_file.dummy_name)
        mock_file.__str__ = Mock(return_value=mock_file.dummy_name)

        actual_hash = remote_file_hash_base64(mock_file)
        assert (
            actual_hash == expected_hash
        ), f"Expected {expected_hash}, but got {actual_hash}"


@pytest.mark.asyncio
@patch.object(
    genai.files.Files,
    "get",
    new_callable=Mock,
)
async def test_wait_for_processing(mock_file_get, monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("GEMINI_API_KEY", "DUMMY")
        dummy_settings = _create_settings()
        gemini_api = GeminiAPI(settings=dummy_settings, log_file=None)
        client = gemini_api._get_client("default")

    # These mocks represent the same file, but at different states/points in time
    starting_file = Mock(
        dummy_name="file1", state=FileState.PROCESSING, sha256_hash="aGFzaDE="
    )

    side_effects = [
        Mock(name="file1", state=FileState.PROCESSING, sha256_hash="aGFzaDE="),
        Mock(name="file1", state=FileState.PROCESSING, sha256_hash="aGFzaDE="),
        Mock(name="file1", state=FileState.ACTIVE, sha256_hash="aGFzaDE="),
        # We should never to this, but including it to differentiate
        # between the function completing because it picked up on the
        # previous file==ACTIVE (correct), or if the function completed
        # because it ran out of side effects (incorrect)
        Mock(name="file1", state=FileState.PROCESSING, sha256_hash="aGFzaDE="),
    ]

    mock_file_get.side_effect = side_effects

    # Call the function to test
    await wait_for_processing(starting_file, client, poll_interval=0)

    # Check that the `get` method was called exactly 3 times
    assert mock_file_get.call_count == 3


@patch.object(
    genai.files.AsyncFiles,
    "list",
    new_callable=AsyncMock,
)
@patch(
    "prompto.apis.gemini.gemini_media.compute_sha256_base64",
    new_callable=MagicMock,
)
def test_upload_media_files_already_uploaded(
    mock_compute_sha256_base64, mock_list_files, monkeypatch, caplog
):
    caplog.set_level(logging.INFO)

    uploaded_file = Mock(dummy_name="remote_uploaded/file1", sha256_hash="aGFzaDE=")
    uploaded_file.configure_mock(name=uploaded_file.dummy_name)
    uploaded_file.__str__ = Mock(return_value=uploaded_file.dummy_name)

    local_file_path = "dummy_local_path/file1.txt"
    expected_log_msg = (
        "File 'dummy_local_path/file1.txt' already uploaded as 'remote_uploaded/file1'"
    )

    with monkeypatch.context() as m:
        m.setenv("GEMINI_API_KEY", "DUMMY")

        mock_compute_sha256_base64.return_value = "aGFzaDE="
        # return_value is a list of a single mock remote file
        mock_list_files.return_value = [uploaded_file]
        dummy_settings = None

        # Pass a list of local file paths to the function
        actual_uploads = upload_media_files([local_file_path], dummy_settings)

        # actual_uploads is a dict of local and remote file names
        assert local_file_path in actual_uploads
        assert actual_uploads[local_file_path] == "remote_uploaded/file1"

        # Check the log message
        assert expected_log_msg in caplog.text


# def test_upload_media_files():
#     pytest.fail("Test not implemented")


# @pytest.mark.asyncio
@patch(
    "prompto.apis.gemini.gemini_media._get_previously_uploaded_files",
    new_callable=AsyncMock,
)
@patch.object(
    genai.files.AsyncFiles,
    "upload",
    new_callable=AsyncMock,
)
@patch(
    "prompto.apis.gemini.gemini_media.compute_sha256_base64",
    new_callable=MagicMock,
)
def test_upload_media_files_new_file(
    mock_compute_sha256_base64,
    mock_files_upload,
    mock_previous_files,
    monkeypatch,
    caplog,
):
    """
    Test the upload_media_files function when the file is not already uploaded, but there are already
    other files uploaded."""
    caplog.set_level(logging.INFO)

    pre_uploaded_file = Mock(
        dummy_name="remote_uploaded/file1",
        sha256_hash=Mock(decode=lambda _: "6861736831"),
    )
    pre_uploaded_file.configure_mock(name=pre_uploaded_file.dummy_name)
    pre_uploaded_file.__str__ = Mock(return_value=pre_uploaded_file.dummy_name)

    previous_files_dict = {
        "hash1": pre_uploaded_file,
    }

    local_file_path = "dummy_local_path/file2.txt"
    expected_log_msgs = [
        "Uploading dummy_local_path/file2.txt to Gemini API",
        "Uploaded file 'remote_uploaded/file2' with hash 'hash2' to Gemini API",
    ]

    new_file = Mock(
        dummy_name="remote_uploaded/file2",
        sha256_hash=Mock(decode=lambda _: "hash2"),
    )
    new_file.configure_mock(name=new_file.dummy_name)
    new_file.__str__ = Mock(return_value=new_file.name)

    with monkeypatch.context() as m:
        m.setenv("GEMINI_API_KEY", "DUMMY")

        mock_compute_sha256_base64.return_value = "hash2"
        mock_previous_files.return_value = previous_files_dict
        mock_files_upload.return_value = new_file

        dummy_settings = None
        actual_uploads = upload_media_files([local_file_path], dummy_settings)

        print(actual_uploads)

        # actual_uploads is a dict of local and remote file names
        assert local_file_path in actual_uploads
        assert actual_uploads[local_file_path] == "remote_uploaded/file2"

        # Check that the previously uploaded file is not in the actual_uploads dict
        assert pre_uploaded_file.dummy_name not in actual_uploads

        # Check the log message
        assert all(msg in caplog.text for msg in expected_log_msgs)


# def test__init_genai():
#     # Is this still required, or is it superseded by the Client object
#     pytest.fail("Test not implemented")


@patch.object(
    genai.files.AsyncFiles,
    "list",
    new_callable=AsyncMock,
)
def test_list_uploaded_files(mock_list_files, caplog, monkeypatch):
    caplog.set_level(logging.INFO)

    # Case 1: No files uploaded
    case_1 = {
        "return_value": [],
        "expected_log_msgs": ["Found 0 files already uploaded at Gemini API"],
    }

    # Case 2: Three files uploaded
    # The example hashes are for the strings "hash1", "hash2", "hash3" eg:
    # >>> "hash1".encode("utf-8").hex()
    # '6861736831'
    case_2 = {
        "return_value": [
            Mock(dummy_name="file1", sha256_hash="aGFzaDE="),
            Mock(dummy_name="file2", sha256_hash="aGFzaDI="),
            Mock(dummy_name="file3", sha256_hash="aGFzaDM="),
        ],
        "expected_log_msgs": [
            "Found 3 files already uploaded at Gemini API",
            "File Name: file1, File Hash: aGFzaDE=",
            "File Name: file2, File Hash: aGFzaDI=",
            "File Name: file3, File Hash: aGFzaDM=",
        ],
    }

    expected_final_log_msg = "All uploaded files listed."

    with monkeypatch.context() as m:
        m.setenv("GEMINI_API_KEY", "DUMMY")

        for case_dict in [case_1, case_2]:

            mocked_list_value = case_dict["return_value"]

            # We need to juggle with the mock names, because we can't set them
            # directly in the constructor. See these docs for more details:
            # https://docs.python.org/3/library/unittest.mock.html#mock-names-and-the-name-attribute
            for mock_file in mocked_list_value:
                mock_file.configure_mock(name=mock_file.dummy_name)
                mock_file.__str__ = Mock(return_value=mock_file.dummy_name)

            mock_list_files.return_value = mocked_list_value

            expected_total_in_log_msg = case_dict["expected_log_msgs"]

            dummy_settings = _create_settings()
            # Call the function to test
            list_uploaded_files(dummy_settings)

            # There is no return value from the function, so we need to check the
            # log messages
            for msg in expected_total_in_log_msg:
                assert msg in caplog.text
            assert expected_final_log_msg in caplog.text


@patch.object(
    genai.files.Files,
    "delete",
    new_callable=Mock,
)
@patch.object(
    genai.files.Files,
    "list",
    new_callable=Mock,
)
def test_delete_uploaded_files(mock_list_files, mock_delete, caplog, monkeypatch):

    caplog.set_level(logging.INFO)

    # Case 1: No files uploaded
    case_1 = []

    # Case 2: Three files uploaded
    # The example hashes are for the strings "hash1", "hash2", "hash3" eg:
    # >>> "hash1".encode("utf-8").hex()
    # '6861736831'
    case_2 = [
        Mock(dummy_name="file1", sha256_hash="aGFzaDE="),
        Mock(dummy_name="file2", sha256_hash="aGFzaDI="),
        Mock(dummy_name="file3", sha256_hash="aGFzaDM="),
    ]
    expected_final_log_msg = "All uploaded files deleted."

    with monkeypatch.context() as m:
        m.setenv("GEMINI_API_KEY", "DUMMY")

        for mocked_list_value in [case_1, case_2]:

            # We need to juggle with the mock names, because we can't set them
            # directly in the constructor. See these docs for more details:
            # https://docs.python.org/3/library/unittest.mock.html#mock-names-and-the-name-attribute
            for mock_file in mocked_list_value:
                mock_file.configure_mock(name=mock_file.dummy_name)
                mock_file.__str__ = Mock(return_value=mock_file.dummy_name)

            mock_list_files.return_value = mocked_list_value

            dummy_settings = _create_settings()
            # Call the function to test
            delete_uploaded_files(dummy_settings)

            # There is no return value from the function, so we need to check the
            # that the delete function was called the expected number of times
            # Add 1 to force it to fail for now.
            assert mock_delete.call_count == len(mocked_list_value)
            assert expected_final_log_msg in caplog.text
