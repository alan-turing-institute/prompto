## Anthropic

**Environment variables**:

* `ANTHROPIC_API_KEY`: the API key for the Anthropic API

**Model-specific environment variables**:

As described in the [model-specific environment variables](./environment_variables.md#model-specific-environment-variables) of the [environment variables document](./environment_variables.md) section, you can set model-specific environment variables for different models in Anthropic by appending the model name to the environment variable name. For example, if `"model_name": "claude-3-haiku-20240307"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:

* `ANTHROPIC_API_KEY_claude_3_haiku_20240307`

Note here we've replaced the `.` and `-` in the model name with underscores `_` to make it a valid environment variable name.

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:

* One of `ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY_model_name`
