# APIs / Models

`prompto` is designed to be extensible and can be used to query different models using different APIs. The library currently supports the following APIs which are grouped into two categories: [_cloud-based services_](#cloud-based-services) and [_self-hosted endpoints_](#self-hosted-endpoints). Cloud-based services refer to LLMs that are hosted by a provider's API endpoint (e.g. OpenAI, Gemini, Anthropic), whereas self-hosted endpoints refer to LLMs that are hosted on a server that you have control over (e.g. Ollama, a Huggingface `text-generation-inference` endpoint).

Note that the names of the APIs are to be used in the `api` key of the `prompt_dict` in the experiment file (see [experiment file documentation](experiment_file.md)) and the names of the models can be specified in the `model_name` key of the `prompt_dict` in the experiment file. The names of the APIs are defined in the `ASYNC_APIS` dictionary in the [`prompto.apis` module](../src/prompto/apis/__init__.py).

## Environment variables

Each API has a number of environment variables that are either required or optional to be set in order to query the model. We recommend setting these environment variables in a `.env` file in the root of the project directory. When you run the `prompto_run_experiment` or `prompto_run_pipeline` commands, the library will look for an `.env` file in the current directory and we then use `python-dotenv` to load the environment variables into the Python environment. An `.env` file would look something like:
```
OPENAI_API_KEY=<YOUR-OPENAI-KEY>
AZURE_OPENAI_API_KEY=<YOUR-AZURE-OPENAI-KEY>
OLLAMA_API_ENDPOINT=<YOUR-OLLAMA_API-ENDPOINT>
```

This allows you to not necessarily need to load them into the global environment in your terminal each time you run an experiment.

Alternatively, you can set the environment variables in your terminal using `export ENVIRONMENT_VARIABLE=value` or having another bash script to set these without a `.env` file - this file would look similar to the example `.env` file, but with `export` in front of each line so that it is loaded into the global environment.

We list the environment variables for each API in the [Implemented APIs](#implemented-apis) section below.

### Model-specific environment variables

There may be some experiments where you wish to query different models from the same API type (e.g. querying different deployments of models from an Azure OpenAI API). In such cases, you can specify the model name in the `model_name` key of the `prompt_dict`.

There may be some cases where the environment variables needed for different models are different, for example if you are querying models which have different API keys (in Azure OpenAI, you may have different models that live on different subscriptions). In such cases, you can set _model-specific environment variables_ by appending the model name to the environment variable name (with a underscore between the environment variable and the model name): `ENVIRONMENT_VARIABLE -> ENVIRONMENT_VARIABLE_model_name`. Note that the model name might have to be formatted to be a valid environment variable name (e.g. replacing `("-", "/", ".", " ")` with underscores `_`).

For example, if you have two models in Azure OpenAI, `model1` and `model2`, you can set the environment variables `AZURE_OPENAI_API_KEY_model1` and `AZURE_OPENAI_API_KEY_model2` with the respective API keys for each model.

The base environment variable name (e.g. `AZURE_OPENAI_API_KEY`) is used as a _default environment variable_ if the model-specific environment variable is not set. They are used either in the case where the model name is not specified in the `prompt_dict` or if the model-specific environment variable is not set.

To clarify, the order of precedence for the API key is as follows:
- If `model_name` is specified in the `prompt_dict`, the model-specific environment variable is used
- If `model_name` is specified in the `prompt_dict` but the model-specific environment variable is not set, the default environment variable is used
- If `model_name` is not specified in the `prompt_dict`, the default environment variable is used
- If neither the model-specific environment variable nor the default environment variable is set, an error is raised

## Cloud-based services

- [Azure OpenAI ("azure-openai")](#azure-openai)
- [OpenAI ("openai")](#openai)
- [Gemini ("gemini")](#gemini)

## Self-hosted endpoints

- [Ollama ("ollama")](#ollama)
- [Huggingface text-generation-inference ("huggingface-tgi")](#huggingface-text-generation-inference)
- [Quart API](#quart-api)

# Implemented APIs

## Azure OpenAI

**Environment variables**:

- `AZURE_OPENAI_API_KEY`: the API key for the Azure OpenAI API
- `AZURE_OPENAI_API_ENDPOINT`: the endpoint for the Azure OpenAI API
- `AZURE_OPENAI_API_VERSION`: the version of the Azure OpenAI API
- `AZURE_OPENAI_MODEL_NAME`: the default model name for the Azure OpenAI API

**Model-specific environment variables**:

As described in the [model-specific environment variables](#model-specific-environment-variables) section, you can set model-specific environment variables for different models in Azure OpenAI by appending the model name to the environment variable name. For example, if `"model_name": "prompto_model"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:
- `AZURE_OPENAI_API_KEY_prompto_model`
- `AZURE_OPENAI_API_ENDPOINT_prompto_model`
- `AZURE_OPENAI_API_VERSION_prompto_model`

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:
- One of `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY_model_name`
- One of `AZURE_OPENAI_API_ENDPOINT` or `AZURE_OPENAI_API_ENDPOINT_model_name`
- `AZURE_OPENAI_MODEL_NAME` if a model is not specified in the `prompt_dict`

## OpenAI

**Environment variables**:

- `OPENAI_API_KEY`: the API key for the OpenAI API
- `OPENAI_MODEL_NAME`: the default model name for the OpenAI API

**Model-specific environment variables**:

As described in the [model-specific environment variables](#model-specific-environment-variables) section, you can set model-specific environment variables for different models in OpenAI by appending the model name to the environment variable name. For example, if `"model_name": "gpt-3.5-turbo"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:
- `OPENAI_API_KEY_gpt_3_5_turbo`

Note here we've replaced the `.` and `-` in the model name with underscores `_` to make it a valid environment variable name.

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:
- One of `OPENAI_API_KEY` or `OPENAI_API_KEY_model_name`
- `OPENAI_MODEL_NAME` if a model is not specified in the `prompt_dict`

## Gemini

**Environment variables**:

- `GEMINI_PROJECT_ID`: the project ID for the Gemini API
- `GEMINI_LOCATION`: the location for the Gemini API
- `GEMINI_MODEL_NAME`: the default model name for the Gemini API

**Model-specific environment variables**:

As described in the [model-specific environment variables](#model-specific-environment-variables) section, you can set model-specific environment variables for different models in Gemini by appending the model name to the environment variable name. For example, if `"model_name": "prompto_model"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:
- `GEMINI_PROJECT_ID_prompto_model`
- `GEMINI_LOCATION_prompto_model`

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:
- `GEMINI_MODEL_NAME` if a model is not specified in the `prompt_dict`
- If you have set up Google Cloud CLI and the project-id or location has not been set, the default project-id and location will be used

## Ollama

See the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs) on how to set up a self-hosted Ollama API endpoint (e.g. using `ollama serve`).

**Environment variables**:

- `OLLAMA_API_ENDPOINT`: the endpoint for the Ollama API
- `OLLAMA_MODEL_NAME`: the default model name for the Ollama API

**Model-specific environment variables**:

As described in the [model-specific environment variables](#model-specific-environment-variables) section, you can set model-specific environment variables for different models in Ollama by appending the model name to the environment variable name. For example, if `"model_name": "prompto_model"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:
- `OLLAMA_API_ENDPOINT_prompto_model`

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:
- One of `OLLAMA_API_ENDPOINT` or `OLLAMA_API_ENDPOINT_model_name`
- `OLLAMA_MODEL_NAME` if a model is not specified in the `prompt_dict`

## Huggingface text-generation-inference

See the [Huggingface `text-generation-inference` repo](https://github.com/huggingface/text-generation-inference) on how to set up a self-hosted Huggingface `text-generation-inference` API endpoint.

**Environment variables**:

- `HUGGINGFACE_TGI_API_ENDPOINT`: the endpoint for the Huggingface `text-generation-inference` API
- `HUGGINGFACE_TGI_API_KEY`: the API key for the Huggingface `text-generation-inference` API

**Model-specific environment variables**:

As described in the [model-specific environment variables](#model-specific-environment-variables) section, you can set model-specific environment variables for different models in Huggingface `text-generation-inference` by appending the model name to the environment variable name.

For example, if you have set up a endpoint for [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) and `"model_name": "flan_t5_xl"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:
- `HUGGINGFACE_TGI_API_ENDPOINT_flan_t5_xl`
- `HUGGINGFACE_TGI_API_KEY_flan_t5_xl`

However, note for the Huggingface `text-generation-inference` API, the model name is only used as an identifier for the pipeline. The model that the endpoint is querying is returned in the response from the API and saved in the output `prompt_dict` in the `"model"` key.
In this case, the completed `prompt_dict` should include the `"model_name": "google/flan-t5-xl"` key-value pair to confirm that the endpoint is indeed querying the correct model.

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:
- One of `HUGGINGFACE_TGI_API_ENDPOINT` or `HUGGINGFACE_TGI_API_ENDPOINT_model_name`

## Quart API

To query models from Huggingface that are not available via the `text-generation-inference` API, we have written a simple [Quart API](../src/prompto/apis/quart/quart_api.py) that can be used to query a text-generation model from the [Huggingface model hub](https://huggingface.co/models) using the Huggingface `transformers` library. This can be started using the `prompto_quart_server` command, e.g.
```
prompto_quart_server --model-name vicgalle/gpt2-open-instruct-v1 --host localhost --port 5000 --max-length 200
```

Once the server is running, you can query the model by sending a POST request to the endpoint with the prompt in the request body, e.g.
```
curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"text": "This is a test prompt"}'
```

In Python, you can use the `requests` library to send a POST request to the endpoint, e.g.
```python
import requests
import json
req = requests.post(
    "http://localhost:5000/generate",
    data=json.dumps({"text": "This is a test prompt"}),
    headers={"Content-Type": "application/json"},
)
```

**Environment variables**:

- `QUART_API_ENDPOINT`: the endpoint for the Quart API

**Model-specific environment variables**:

As described in the [model-specific environment variables](#model-specific-environment-variables) section, you can set model-specific environment variables for different models in the Quart API by appending the model name to the environment variable name. For example, if `"model_name": "vicgalle/gpt2-open-instruct-v1"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:
- `QUART_API_ENDPOINT_vicgalle_gpt2_open_instruct_v1`

Similarly to the Huggingface `text-generation-inference` API, the model name is only used as an identifier for the pipeline. The model that the endpoint is querying is returned in the response from the API and saved in the output `prompt_dict` in the `"model"` key.
In this case, the completed `prompt_dict` should include the `"model_name": "vicgalle/gpt2-open-instruct-v1"` key-value pair to confirm that the endpoint is indeed querying the correct model.

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:
- One of `QUART_API_ENDPOINT` or `QUART_API_ENDPOINT_model_name`
