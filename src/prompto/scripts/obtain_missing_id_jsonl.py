import argparse
import json

from tqdm import tqdm


def get_ids(file: str, id_name: str = "id") -> list[str]:
    """
    Loop through the jsonl file and return a list of ids.

    Parameters
    ----------
    file : str
        The path to the jsonl file
    id_name : str, optional
        The name of the id field in the jsonl file,
        by default "id"

    Returns
    -------
    list[str]
        A list of ids
    """
    ids = []
    n_lines = sum(1 for _ in open(file, "r"))
    with open(file, "r") as f:
        for line in tqdm(
            f, desc="Reading jsonl file to get ids", unit="lines", total=n_lines
        ):
            try:
                data = json.loads(line)
                ids.append(data[id_name])
            except:
                print(f"Error reading line: {line}")

    return ids


def obtain_missing_jsonl(
    input_file: str, output_file: str, new_experiment_file: str, id_name: str = "id"
) -> None:
    """
    Loops through the input_file and checks if the id is in the output_file.
    If it is not, then it adds the line to the new_experiment_file.

    Parameters
    ----------
    input_file : str
        Path to input jsonl experiment file with prompts
    output_file : str
        Path to output jsonl file with prompts
    new_experiment_file : str
        Path to new jsonl experiment file with prompts
        that were missing in the output_file
    id_name : str, optional
        The name of the id field in the jsonl file,
        by default "id"
    """
    output_file_ids = get_ids(
        file=output_file,
        id_name=id_name,
    )
    added = 0
    n_lines = sum(1 for _ in open(input_file, "r"))
    with open(input_file, "r") as f:
        for line in tqdm(
            f,
            desc="Reading input file to get missing prompts",
            unit="lines",
            total=n_lines,
        ):
            data = json.loads(line)
            if data[id_name] not in output_file_ids:
                # write this line to new_experiment_file
                with open(new_experiment_file, "a") as f:
                    f.write(line)

                added += 1

    if added == 0:
        print("No missing prompts found")
    else:
        print(f"Added {added} missing prompts to {new_experiment_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="input jsonl experiment file with prompts",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="output jsonl file with prompts",
    )
    parser.add_argument(
        "--new",
        "-n",
        type=str,
        required=True,
        help="jsonl file with prompts",
    )
    parser.add_argument(
        "--id",
        type=str,
        default="id",
        help="name of the id field in the jsonl file",
    )
    args = parser.parse_args()
    obtain_missing_jsonl(
        input_file=args.input,
        output_file=args.output,
        new_experiment_file=args.new,
        id_name=args.id,
    )


if __name__ == "__main__":
    main()
