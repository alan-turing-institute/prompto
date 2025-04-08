import json
from argparse import Namespace

import pytest
from cli_test_helpers import ArgvContext, shell

from prompto.apis import ASYNC_APIS
from prompto.upload_media import (
    UPLOAD_APIS,
    _read_experiment_file,
    _resolve_output_file_location,
    check_uploads_by_api,
    update_experiment_file,
)


def test_upload_media_no_inputs():
    result = shell("prompto_upload_media")
    assert result.exit_code != 0
    assert "usage:" in result.stderr


def test_supported_apis_list():
    """
    Test that all the apis listed in `UPLOAD_APIS` are part of the
    `ASYNC_APIS` dictionary. Protects against typos in the list of supported APIs.
    """
    for api in UPLOAD_APIS:
        assert api in ASYNC_APIS, f"API {api} not found in ASYNC_APIS"


def test_upload_media_supported_apis():
    """
    Test that the upload_media command creates a meaningful error if the user
    attempts to upload media to an unsupported API.
    """

    # Passing case (eg 'gemini')
    assert check_uploads_by_api("gemini")

    # Failing case (eg 'openai')
    with pytest.raises(NotImplementedError) as nie:
        check_uploads_by_api("openai")

    assert str(nie.value) == "Uploading media files to openai is not supported yet."


def test_output_file_options(tmp_path):

    input_file = tmp_path / "test_file.jsonl"
    expected_default_output_file = str(tmp_path / "test_file_uploaded.jsonl")

    # Test the default output file location
    args = Namespace(
        file=str(input_file),
        output_file=None,
        overwrite_output=False,
    )
    _resolve_output_file_location(args)
    assert args.output_file == expected_default_output_file

    args = Namespace(
        file=str(input_file),
        output_file=None,
        overwrite_output=True,
    )
    _resolve_output_file_location(args)
    assert args.output_file == expected_default_output_file

    # Test the specified output file location
    non_default_output_file = tmp_path / "output_file.jsonl"
    args = Namespace(
        file=str(input_file),
        output_file=str(non_default_output_file),
        overwrite_output=False,
    )
    _resolve_output_file_location(args)
    assert args.output_file == str(non_default_output_file)

    # Create the output file and check that it raises an error if overwrite is not set
    non_default_output_file.touch()
    args = Namespace(
        file=str(input_file),
        output_file=str(non_default_output_file),
        overwrite_output=False,
    )
    # Check that it raises an error because the output file already exists
    with pytest.raises(ValueError, match="overwrite") as ve:
        _resolve_output_file_location(args)

    # Now check that it works with overwrite set to True
    args = Namespace(
        file=str(input_file),
        output_file=str(non_default_output_file),
        overwrite_output=True,
    )
    _resolve_output_file_location(args)
    assert args.output_file == str(non_default_output_file)


def test_read_experiment_file(tmp_path):
    # Create a temporary experiment file
    experiment_file = tmp_path / "experiment.jsonl"
    media_dir = tmp_path / "media"
    media_dir.mkdir(exist_ok=True)

    with open(experiment_file, "w") as f:
        # Note the third line should pass. It is not a supported API, but it doesn't actually
        # contain any media files to upload.
        f.write(
            '{"api": "gemini", "prompt": [{"parts": [{"media": "image1.jpg"}]}]}\n'
            '{"api": "gemini", "prompt": [{"parts": [{"media": "image2.jpg"}]}]}\n'
            '{"api": "openai", "prompt": [{"parts": "This is a prompt without an image"}]}'
        )

    # Call the function
    actual_media_files, actual_prompt_dict_list = _read_experiment_file(str(experiment_file), str(media_dir))

    # Check the media file results
    expected_media_files = {
        str(media_dir / "image1.jpg"),
        str(media_dir / "image2.jpg"),
    }

    assert actual_media_files == expected_media_files

    # Check the prompt dictionary results
    assert len(actual_prompt_dict_list) == 3
    assert actual_prompt_dict_list[0]["api"] == "gemini"
    assert actual_prompt_dict_list[0]["prompt"][0]["parts"][0]["media"] == "image1.jpg"
    assert actual_prompt_dict_list[1]["api"] == "gemini"
    assert actual_prompt_dict_list[1]["prompt"][0]["parts"][0]["media"] == "image2.jpg"
    assert actual_prompt_dict_list[2]["api"] == "openai"


def test_update_experiment_file(tmp_path):
    # Create a temporary experiment file
    output_file = tmp_path / "test_experiment.jsonl"
    media_dir = tmp_path / "media"
    media_dir.mkdir(exist_ok=True)
    media_dir = str(media_dir)

    prompt_dict_list = [
        {"api": "gemini", "prompt": [{"parts": [{"media": "image1.jpg"}]}]},
        {"api": "gemini", "prompt": [{"parts": [{"media": "image2.jpg"}]}]},
        {"api": "openai", "prompt": [{"parts": "This is a prompt without an image"}]},
    ]

    uploaded_files = {
        f"{media_dir}/image1.jpg": "uploaded_image1.jpg",
        f"{media_dir}/image2.jpg": "uploaded_image2.jpg",
    }

    # Call the function
    update_experiment_file(
        prompt_dict_list,
        uploaded_files,
        str(output_file),
        media_dir,
    )

    with open(output_file, "r") as output_file:
        # No need to parse the lines as json.
        # For assertions, we can just use the raw strings
        actual_output1 = output_file.readline()
        actual_output2 = output_file.readline()
        actual_output3 = output_file.readline()

        assert "uploaded_filename" in actual_output1
        assert "uploaded_image1.jpg" in actual_output1
        assert "uploaded_filename" in actual_output2
        assert "uploaded_image2.jpg" in actual_output2
        # No uploaded filename for the third line
        assert "uploaded_filename" not in actual_output3
