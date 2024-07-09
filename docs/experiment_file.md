# Setting up an experiment file

An experiment file is a [JSON Lines (jsonl)](https://jsonlines.org/) file that contains the prompts for the experiments along with any other parameters or metadata that is required for the prompt. Each line in the jsonl file is a valid JSON value which defines a particular input to the LLM which we will obtain a response for. We often refer to a single line in the jsonl file as a "`prompt_dict`" (prompt dictionary).

For all models/APIs, we require the following keys in the `prompt_dict`:
- `prompt`: the prompt for the model
    - This is typically a _string_ that is passed to the model to generate a response, but for certain APIs and models, this could also take different forms. For example, for some API endpoints (e.g. OpenAI (`"api": "openai"`)) the prompt could also be a list of strings in which case we consider this to be a sequence of prompts to be sent to the model, or it could be a list of dictionaries where each dictionary has a "role" and "content" key which can be used to define a history of a conversation which is sent to the model for a response.
    - See the [documentation](models.md) for the specific APIs/models for more details on the different accepted formats of the prompt.
- `api`: the name of the API to query
    - See the [available APIs/models](models.md) for the list of supported APIs and the corresponding names to use in the `api` key
    - They are defined in the `ASYNC_APIS` dictionary in the [`prompto.apis` module](../src/prompto/apis/__init__.py)
- `model_name`: the name of the model to query
    - For most API endpoints, it is possible to define the name of the model to query. For example, for the OpenAI API (`"api": "openai"`), the model name could be `"gpt-3.5-turbo"`, `"gpt-4"`, etc.

In addition, there are other optional keys that can be included in the `prompt_dict`:
- `parameters`: the parameter settings / generation config for the query (given as a dictionary)
    - This is a dictionary that contains the parameters for the query. The parameters are specific to the model and the API being used. For example, for the Gemini API (`"api": "gemini"`), some paramters to configure are {`temperature`, `max_output_tokens`, `top_p`, `top_k`} etc. which are used to control the generation of the response. For the OpenAI API (`"api": "openai"`), some of these parameters are named differently for instance the maximum output tokens is set using the `max_tokens` parameter and `top_k` is not available to set. For Ollama (`"api": "ollama"`), the parameters are different again, e.g. the maximum number of tokens to predict is set using `num_predict`
    - See the API documentation for the specific API for the list of parameters that can be set and their default values
- `group`: a user-specified grouping of the prompts
    - This is a string that can be used to group the prompts together. This is useful when you want to process groups of prompts in parallel (e.g. when using the `--parallel` flag in the pipeline)
    - Note that you can use parallel processing without using the "group" key, but using this key allows you to have full control in order group the prompts in a way that makes sense for your use case. See the [specifying rate limits documentation](rate_limits.md) for more details on parallel processing

Lastly, there are other optional keys that are only available for certain APIs/models. For example, for the Gemini API, you can have a `multimedia` key which is a list of dictionaries defining the multimedia files (e.g. images/videos) to be used in the prompt to a multimodal LLM. For these, see the documentation for the specific API/model for more details.
