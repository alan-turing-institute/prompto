# Models

- Azure OpenAI
    - Need to set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_ENDPOINT` environment variables. You can also set the `AZURE_OPENAI_API_VERSION` variable too. Also recommended to set the `AZURE_OPENAI_MODEL_ID` in the environment variable to either avoid passing in the `model_name` each time if using the same one consistently.
- OpenAI
    - Need to set `OPENAI_API_KEY` environment variable. Also recommended to set the `OPENAI_MODEL_NAME` in the environment variable to either avoid passing in the `model_name` each time if using the same one consistently.
- Gemini
    - Need to set `GEMINI_PROJECT_ID`, and `GEMINI_LOCATION` environment variables. Also recommended to set the `GEMINI_MODEL_NAME` in the environment variable to either avoid passing in the `model_name` each time if using the same one consistently.
- Ollama endpoints
- Huggingface `text-generation-inference` endpoints
- Simple Quart API endpoints
