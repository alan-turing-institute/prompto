## Vertex AI

**Environment variables**:

* `VERTEXAI_PROJECT_ID`: the project ID for the Gemini API
* `VERTEXAI_LOCATION_ID`: the location for the Gemini API

**Model-specific environment variables**:

As described in the [model-specific environment variables](./environment_variables.md#model-specific-environment-variables) of the [environment variables document](./environment_variables.md) section, you can set model-specific environment variables for different models in Gemini by appending the model name to the environment variable name. For example, if `"model_name": "prompto_model"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:

* `VERTEXAI_PROJECT_ID_prompto_model`
* `VERTEXAI_LOCATION_ID_prompto_model`

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:

* If you have set up Google Cloud CLI and a default project-id or location have been set, the default project-id and location will be used. In this setting, the `VERTEXAI_PROJECT_ID` and `VERTEXAI_LOCATION_ID` environment variables are optional.
