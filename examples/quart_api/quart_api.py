import argparse

import torch
from quart import Quart, jsonify, request
from transformers import pipeline

# Parsing command-line arguments
parser = argparse.ArgumentParser(description="Run the text generation API")
parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    required=True,
    help="Huggingface model name (e.g., 'vicgalle/gpt2-open-instruct-v1') to be used in a transformers pipeline",
)
parser.add_argument(
    "-l",
    "--max_length",
    type=int,
    default=200,
    help="Maximum length of the generated text",
)
parser.add_argument(
    "-p", "--port", type=int, default=5000, help="Port number on which to run the API"
)

args = parser.parse_args()

app = Quart(__name__)

# Check if MPS (Apple Silicon GPU) is available, else use CUDA if available, else CPU
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

try:
    pipe = pipeline("text-generation", model=args.model_name)
except OSError as exc:
    raise OSError(f"Model '{args.model_name}' not found") from exc


@app.route("/generate", methods=["POST"])
async def generate():
    data = await request.get_json()
    model_key = data.get("model")

    if model_key != args.model_name:
        return jsonify(
            {
                "error": f"Model '{model_key}' not found. Please use model '{args.model_name}'"
            }
        )
    text = data.get("text")

    # generate output using pipeline
    response = pipe(text, max_length=args.max_length)

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, port=args.port)
