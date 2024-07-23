# APIs / Models

`prompto` is designed to be extensible and can be used to query different models using different APIs. The library currently supports the following APIs which are grouped into two categories: [_cloud-based services_](#cloud-based-services) and [_self-hosted endpoints_](#self-hosted-endpoints). Cloud-based services refer to LLMs that are hosted by a provider's API endpoint (e.g. OpenAI, Gemini, Anthropic), whereas self-hosted endpoints refer to LLMs that are hosted on a server that you have control over (e.g. Ollama, a Huggingface `text-generation-inference` endpoint).

Note that the names of the APIs are to be used in the `api` key of the `prompt_dict` in the experiment file (see [experiment file documentation](experiment_file.md)) and the names of the models can be specified in the `model_name` key of the `prompt_dict` in the experiment file. The names of the APIs are defined in the `ASYNC_APIS` dictionary in the [`prompto.apis` module](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/apis/__init__.py).

In Python, you can see which APIs you have available to you by running the following code:

```python
from prompto.apis import ASYNC_APIS
print(ASYNC_APIS.keys())
```

Note that you need to have the correct dependencies installed to be able to use the APIs. See the [installation guide](../README.md#installation) for more details on how to install the dependencies for the different APIs.

## Environment variables

Each API has a number of environment variables that are either required or optional to be set in order to query the model. See the [environment variables documentation](environment_variables.md) for more details on how to set these environment variables.

## Cloud-based services

* [Azure OpenAI ("azure-openai")](./azure_openai.md)
* [OpenAI ("openai")](./openai.md)
* [Anthropic ("anthropic")](./anthropic.md)
* [Gemini ("gemini")](./gemini.md)
* [Vertex AI ("vertexai")](./vertexai.md)

## Self-hosted endpoints

* [Ollama ("ollama")](./ollama.md)
* [Huggingface text-generation-inference ("huggingface-tgi")](./huggingface_tgi.md)
* [Quart API](./quart.md)
