import argparse
import logging
import os

from PIL import Image
from tqdm import tqdm


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        "-f",
        help="Folder with media files",
        type=str,
    )
    args = parser.parse_args()

    # initialise logging
    logging.basicConfig(
        datefmt=r"%Y-%m-%d %H:%M:%S",
        format="%(asctime)s [%(levelname)8s] %(message)s",
        level=logging.INFO,
    )

    for item in tqdm(os.listdir(args.folder)):
        file = os.path.join(args.folder, item)
        if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
            image = Image.open(file)
            converted_image = image.convert("RGB")
            converted_image.save(file, "JPEG")
        elif file.lower().endswith(".png"):
            image = Image.open(file)
            converted_image = image.convert("RGB")
            converted_image.save(file, "PNG")
        else:
            logging.info(f"Skipping {file} as it is not a supported file type")

    logging.info("Finished converting images")


if __name__ == "__main__":
    main()
