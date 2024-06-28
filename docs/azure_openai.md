## Azure OpenAI

**Environment variables**:

* `AZURE_OPENAI_API_KEY`: the API key for the Azure OpenAI API
* `AZURE_OPENAI_API_ENDPOINT`: the endpoint for the Azure OpenAI API
* `AZURE_OPENAI_API_VERSION`: the version of the Azure OpenAI API

**Model-specific environment variables**:

As described in the [model-specific environment variables](./environment_variables.md#model-specific-environment-variables) section, you can set model-specific environment variables for different models in Azure OpenAI by appending the model name to the environment variable name. For example, if `"model_name": "prompto_model"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:

* `AZURE_OPENAI_API_KEY_prompto_model`
* `AZURE_OPENAI_API_ENDPOINT_prompto_model`
* `AZURE_OPENAI_API_VERSION_prompto_model`

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:

* One of `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY_model_name`
* One of `AZURE_OPENAI_API_ENDPOINT` or `AZURE_OPENAI_API_ENDPOINT_model_name`
