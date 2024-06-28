## Huggingface text-generation-inference

See the [Huggingface `text-generation-inference` repo](https://github.com/huggingface/text-generation-inference) on how to set up a self-hosted Huggingface `text-generation-inference` API endpoint.

**Environment variables**:

* `HUGGINGFACE_TGI_API_ENDPOINT`: the endpoint for the Huggingface `text-generation-inference` API
* `HUGGINGFACE_TGI_API_KEY`: the API key for the Huggingface `text-generation-inference` API

**Model-specific environment variables**:

As described in the [model-specific environment variables](./environment_variables.md#model-specific-environment-variables) of the [environment variables document](./environment_variables.md) section, you can set model-specific environment variables for different models in Huggingface `text-generation-inference` by appending the model name to the environment variable name.

For example, if you have set up a endpoint for [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) and `"model_name": "flan_t5_xl"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:

* `HUGGINGFACE_TGI_API_ENDPOINT_flan_t5_xl`
* `HUGGINGFACE_TGI_API_KEY_flan_t5_xl`

However, note for the Huggingface `text-generation-inference` API, the model name is only used as an identifier for the pipeline. The model that the endpoint is querying is returned in the response from the API and saved in the output `prompt_dict` in the `"model"` key.
In this case, the completed `prompt_dict` should include the `"model_name": "google/flan-t5-xl"` key-value pair to confirm that the endpoint is indeed querying the correct model.

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:

* One of `HUGGINGFACE_TGI_API_ENDPOINT` or `HUGGINGFACE_TGI_API_ENDPOINT_model_name`
