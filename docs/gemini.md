## Gemini

**Environment variables**:

* `GEMINI_API_KEY`: the project ID for the Gemini API

**Model-specific environment variables**:

As described in the [model-specific environment variables](./environment_variables.md#model-specific-environment-variables) of the [environment variables document](./environment_variables.md) section, you can set model-specific environment variables for different models in Gemini by appending the model name to the environment variable name. For example, if `"model_name": "prompto_model"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:

* `GEMINI_API_KEY_prompto_model`

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:

* One of `GEMINI_API_KEY` or `GEMINI_API_KEY_model_name`
