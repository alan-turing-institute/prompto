import argparse

import torch
from quart import Quart, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

# Parsing command-line arguments
parser = argparse.ArgumentParser(description="Run the text generation API")
parser.add_argument(
    "-m", "--model_name", type=str, required=True, help="Model name (e.g., 'gpt2')"
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

# Load models
models = {}

# GPT2
models["gpt2"] = {
    "tokenizer": AutoTokenizer.from_pretrained("vicgalle/gpt2-open-instruct-v1"),
    "model": AutoModelForCausalLM.from_pretrained("vicgalle/gpt2-open-instruct-v1").to(
        device
    ),
}

if args.model_name not in models:
    raise ValueError(f"Model '{args.model_name}' not found")


@app.route("/generate", methods=["POST"])
async def generate():
    data = await request.get_json()
    model_key = data.get("model")
    text = data.get("text")

    tokenizer = models[model_key]["tokenizer"]
    model = models[model_key]["model"]

    # Encode the text input and generate response
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=args.max_length)

    # Decode the generated tokens to text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response_text})


if __name__ == "__main__":
    app.run(debug=True, port=args.port)
