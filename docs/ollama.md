## Ollama

See the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs) on how to set up a self-hosted Ollama API endpoint (e.g. using `ollama serve`).

**Environment variables**:

* `OLLAMA_API_ENDPOINT`: the endpoint for the Ollama API

**Model-specific environment variables**:

As described in the [model-specific environment variables](./environment_variables.md#model-specific-environment-variables) of the [environment variables document](./environment_variables.md) section, you can set model-specific environment variables for different models in Ollama by appending the model name to the environment variable name. For example, if `"model_name": "prompto_model"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:

* `OLLAMA_API_ENDPOINT_prompto_model`

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:

* One of `OLLAMA_API_ENDPOINT` or `OLLAMA_API_ENDPOINT_model_name`
