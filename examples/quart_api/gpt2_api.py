import torch
from quart import Quart, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer

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


@app.route("/generate", methods=["POST"])
async def generate():
    data = await request.get_json()
    model_key = data.get("model")
    text = data.get("text")

    if model_key not in models:
        return jsonify({"error": "Model not found"}), 404

    tokenizer = models[model_key]["tokenizer"]
    model = models[model_key]["model"]

    # Encode the text input and generate response
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=200)

    # Decode the generated tokens to text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response_text})


if __name__ == "__main__":
    app.run(debug=True)
