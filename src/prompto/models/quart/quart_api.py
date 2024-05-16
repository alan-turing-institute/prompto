import argparse
import os

import torch
from quart import Quart, jsonify, request
from transformers import pipeline


def main():
    # parsing command-line arguments
    parser = argparse.ArgumentParser(description="Run the text generation API")
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        required=True,
        help="Huggingface model name (e.g., 'vicgalle/gpt2-open-instruct-v1') to be used in a transformers pipeline",
    )
    parser.add_argument(
        "-l",
        "--max-length",
        type=int,
        default=200,
        help="Maximum length of the generated text",
    )
    parser.add_argument(
        "--pipeline-task",
        "-t",
        help="task defining the HF pipeline",
        type=str,
        default="text-generation",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host on which to run the API",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port number on which to run the API",
    )

    args = parser.parse_args()

    app = Quart(__name__)

    # check if MPS (Apple Silicon GPU) is available, else use CUDA if available, else CPU
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        pipe = pipeline(
            task=args.pipeline_task,
            model=args.model_name,
            device_map=device,
            token=os.environ.get("HUGGINGFACE_TOKEN"),
            return_full_text=False,
        )
    except OSError as exc:
        raise OSError(
            f"Model '{args.model_name}' not found, or maybe you might need to set your HUGGINGFACE_TOKEN environment variable"
        ) from exc
    except Exception as exc:
        raise Exception(f"Error loading model '{args.model_name}'") from exc

    @app.get("/")
    async def ping():
        return "pong"

    @app.route("/generate", methods=["POST"])
    async def generate():
        data = await request.get_json()
        text = data.get("text")
        # generate output using pipeline
        response = pipe(text, max_length=args.max_length)

        return jsonify({"response": response, "model": args.model_name})

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
