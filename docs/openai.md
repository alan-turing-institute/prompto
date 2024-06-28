## OpenAI

**Environment variables**:

* `OPENAI_API_KEY`: the API key for the OpenAI API

**Model-specific environment variables**:

As described in the [model-specific environment variables](./environment_variables.md#model-specific-environment-variables) of the [environment variables document](./environment_variables.md) section, you can set model-specific environment variables for different models in OpenAI by appending the model name to the environment variable name. For example, if `"model_name": "gpt-3.5-turbo"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:

* `OPENAI_API_KEY_gpt_3_5_turbo`

Note here we've replaced the `.` and `-` in the model name with underscores `_` to make it a valid environment variable name.

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:

* One of `OPENAI_API_KEY` or `OPENAI_API_KEY_model_name`
