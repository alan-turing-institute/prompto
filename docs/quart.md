## Quart API

To query models from Huggingface that are not available via the `text-generation-inference` API, we have written a simple [script to start up a Quart API](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/apis/quart/quart_api.py)
``` that can be used to query a text-generation model from the [Huggingface model hub](https://huggingface.co/models) using the Huggingface `transformers` library. This can be started using the `prompto_quart_server` command, e.g.

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

* `QUART_API_ENDPOINT`: the endpoint for the Quart API

**Model-specific environment variables**:

As described in the [model-specific environment variables](./environment_variables.md#model-specific-environment-variables) of the [environment variables document](./environment_variables.md) section, you can set model-specific environment variables for different models in the Quart API by appending the model name to the environment variable name. For example, if `"model_name": "vicgalle/gpt2-open-instruct-v1"` is specified in the `prompt_dict`, the following model-specific environment variables can be used:

* `QUART_API_ENDPOINT_vicgalle_gpt2_open_instruct_v1`

Similarly to the Huggingface `text-generation-inference` API, the model name is only used as an identifier for the pipeline. The model that the endpoint is querying is returned in the response from the API and saved in the output `prompt_dict` in the `"model"` key.
In this case, the completed `prompt_dict` should include the `"model_name": "vicgalle/gpt2-open-instruct-v1"` key-value pair to confirm that the endpoint is indeed querying the correct model.

**Required environment variables**:

For any given `prompt_dict`, the following environment variables are required:

* One of `QUART_API_ENDPOINT` or `QUART_API_ENDPOINT_model_name`
