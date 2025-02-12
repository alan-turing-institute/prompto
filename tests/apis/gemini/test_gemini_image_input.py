from prompto.apis.gemini.gemini_utils import parse_parts_value
from PIL import Image
import os


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


def test_parse_parts_value_image_dict():
    part = {"type": "image", "media": "pantani_giro.jpg"}
    media_folder = "media"

    # Create a mock image
    if not os.path.exists(media_folder):
        os.makedirs(media_folder)
    image_path = os.path.join(media_folder, "pantani_giro.jpg")
    image = Image.new("RGB", (100, 100), color="red")
    image.save(image_path)

    result = parse_parts_value(part, media_folder)

    # Assert the result
    assert result.mode == "RGB"
    assert result.size == (100, 100)
    assert result.filename.endswith("pantani_giro.jpg")

    # Clean up the mock image
    os.remove(image_path)
    os.rmdir(media_folder)


def test_parse_parts_value_video():
    part = {"type": "video", "media": "pantani_giro.mp4"}
    media_folder = "media"

    result = parse_parts_value(part, media_folder)
    assert result == part
