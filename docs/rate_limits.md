# Specifying rate limits

When running the pipeline or an experiment, there are certain settings to define how to run the experiments which are described in the [pipeline documentation](./pipeline.md#pipeline-settings). These can be set using the command line interfaces.

One of the key settings is the rate limit which is the maximum number of queries that can be sent to an API/model within a minute. This is important to prevent the API from being overloaded and to prevent the user from being blocked by the API. The (default) rate limit can be set using the `--max-queries` or `-m` flag. By default, the rate limit is set to `10` queries per minute.

Another key setting is whether or not to process the prompts in the experiments in parallel meaning that we send the queries to the different APIs (which typically have separate and independent rate limits) in parallel. This can be set using the `--parallel` or `-p` flag. In this document, we will describe how to set the rate limits for each API or group of APIs when the `--parallel` flag is set and how to use the `--max-queries-json` or `-mqj` flag to do this.

For more examples and a walkthrough of how to set the rate limits for parallel processing of prompts, see the [Grouping prompts and specifying rate limits notebook](../examples/notebooks/grouping_prompts_and_specifying_rate_limits.ipynb).

## Using no parallel processing

If the `--parallel` flag is not set, the rate limit is set using the `--max-queries` flag. This is the simplest pipeline setting and typically should only be used when the experiment file contains prompts for a single API and model, e.g.:
```json
{"id": 0, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of France?"}
{"id": 1, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of Germany?"}
```

In this case, there is only one model to query through the same API and so parallel processing is not necessary. The rate limit can be set using the `--max-queries` flag, e.g. to send 5 per minute (the default is 10):
```bash
prompto_run_experiment --file path/to/experiment.jsonl --data-folder data --max-queries 5
```

## Using parallel processing

When the `--parallel` flag is set, we will always try to perform a grouping of the prompts and we group prompts according to the "group" or "api" key in the `prompt_dict` ([line in the jsonl experiment file](./experiment_file.md)) and the (optional) user-specified rate limits for each group or API are set using the `--max-queries-json` or `-mqj` flag. In this section, we will detail how this grouping is performed. We provide a few different scenarios for splitting the prompts into different queues and setting the rate limits for parallel processing of them. More examples can also be found in the [Grouping prompts and specifying rate limits notebook](../examples/notebooks/grouping_prompts_and_specifying_rate_limits.ipynb).

When using the `--max-queries-json` flag, you must pass a path to a json file which contains the maximum number of queries to send within a minute for each API, model or group. In this json, the keys are API names (e.g. "openai", "gemini", etc.) or group names and the values can either be integers which represent the corresponding rate limit for the API or group. The values can also themselves be another dictionary where keys are model names and values are integers representing the rate limit for that model. If the json file is not provided, the `--max-queries` value is used for all APIs or groups.

To summarise, the json file should have the following structure:
- The keys are the API names or group names
- The values can either be:
    - integers which represent the corresponding rate limit for the API or group
    - another dictionary where keys are model names and values are integers representing the rate limit for that model

Concretely, the json file should look like this:
```json
{
    "api-1": 10,
    "api-2": {
        "default": 20,
        "model-1": 15,
        "model-2": 25

    },
    "group-1": 5,
    "group-2": {
        "model-1": 15,
        "model-2": 25
    }
}
```

In the codebase, this json defines the `max_queries_dict` which is a dictionary which defines the rate limits to set for different groups of prompts. We use this dictionary to generate several different _groups/queues of prompts_ which are then processed in parallel. Note that this dictionary/json is only used to specify any rate limits which are _different_ from the default rate limit which is set using the `--max-queries` flag. Anything that is not specified in the json file will be set to the default rate limit.

When the `--parallel` flag is set, we will always try to perform a grouping of the prompts based on first the "group" key and then the "api" key. If there is a "model_name" key and the model name has been specified in the `max_queries_dict` for the group or API (by having a sub-dictionary as a value to the group or API name), then the prompt is assigned to the model-specific queue for that group or API.

In particular, we use the `max_queries_dict` and loop through the `prompt_dicts` in the experiment file to determine which group/queue the prompt belongs to. When deciding this, the following hierarchy is used:
1. If the prompt has a "group" key, then the prompt is assigned to the group defined by the value of the "group" key.
    - If the prompt has a "model_name" key, and this model name has been specified in the `max_queries_dict`, then the prompt is assigned to the group defined by the `{group}-{model_name}`
2. If the prompt has an "api" key, then the prompt is assigned to the group defined by the value of the "api" key.
    - If the prompt has a "model_name" key, and this model name has been specified in the `max_queries_dict`, then the prompt is assigned to the group defined by the `{api}-{model_name}`

By first looking for a "group" key, this allows the user to have full control over how the prompts are split into different groups/queues.

Below we detail a few different scenarios for splitting the prompts into different queues and setting the rate limits for parallel processing of them. There are different levels of granularity and user-control that can be used to set for the rate limits:
- [Same rate limit for all APIs (max_queries_dict is not provided)](#same-rate-limit-for-all-apis)
- [Different rate limits for each API type](#different-rate-limits-for-each-api-type)
- [Different rate limits for each API type and model](#different-rate-limits-for-each-api-type-and-model)
- [Full control: Using the "groups" key to define user-specified groups of prompts](#full-control-using-the-groups-key-to-define-user-specified-groups-of-prompts)

### Same rate limit for all APIs

If the `--parallel` flag is set but the `--max-queries-json` flag is not used, then this is is equivalent to setting the same rate limit for all API types that are present in the experiment file. This is the simplest case of parallel processing and is useful when the experiment file contains prompts for different APIs but we want to set the same rate limit for all of them.

For example, consider the following experiment file:
```json
{"id": 0, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of France?"}
{"id": 1, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of Germany?"}
{"id": 2, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of France?"}
{"id": 3, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of Germany?"}
{"id": 4, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of France?"}
{"id": 5, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of Germany?"}
{"id": 6, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of France?"}
{"id": 7, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of Germany?"}
{"id": 8, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of France?"}
{"id": 9, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of Germany?"}
{"id": 10, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of France?"}
{"id": 11, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of Germany?"}
```

As noted above, since there are no "group" keys in the experiment file, the prompts are simply grouped by the "api" key.

If `--parallel` flag is used but no `max_queries_dict` is provided (i.e. the `--max-queries-json` flag is not used in the CLI), then we simply group the prompts by the "api" key and send the prompts to the different APIs in parallel with the same rate limit, e.g.:
```bash
prompto_run_experiment \
    --file path/to/experiment.jsonl \
    --data-folder data \
    --max-queries 5 \
    --parallel
```

In this case, three groups/queues of prompts are created: one for the "gemini" API, one for the "openai" API and one for the "ollama" API. The rate limit of 5 queries per minute is applied to both groups since we specified `--max-queries 5`.

### Different rate limits for each API type

To build on the above example, if we want to set different rate limits for each API type, we can use the `--max-queries-json` flag where the keys of the json file are the API names and the values are the rate limits for each API. For example, consider the following json file `max_queries.json`:
```json
{
    "openai": 20,
    "gemini": 10
}
```

Then we can run the experiment with the following command:
```bash
prompto_run_experiment \
    --file path/to/experiment.jsonl \
    --data-folder data \
    --max-queries 5 \
    --max-queries-json max_queries.json \
    --parallel
```

In this case, three groups/queues of prompts are created: one for the "gemini" API, one for the "openai" API and one for the "ollama" API. The rate limit of 10 queries per minute is applied to the "gemini" group, the rate limit of 20 queries per minute is applied to the "openai" group and since we did not specify a rate limit for "ollama", they are sent to the endpoint at the default 5 per minute rate established by the `--max-queries 5` flag in the command.

It is important to note that the keys in the json file must match the values of the "api" key in the experiment file. If there is an API in the experiment file that is not in the json file, then the rate limit for that API will be set to the default rate limit which is set using the `--max-queries` flag.
If we had accidentally misspelled "openai" as "openaii" in the json file, then the rate limit for the "openai" prompts would have been set to the default rate.
The reason why we do not have a check on the spelling is since we allow for user-specified grouping of prompts which we discuss in the [full control section](#full-control-using-the-groups-key-to-define-user-specified-groups-of-prompts).

### Different rate limits for each API type and model

For some APIs, there are different models which can be queried which may have different rate limits. As noted above, the values of the json file can themselves be another dictionary where keys are model names and values are integers representing the rate limit for that model. This allows us to have further control on the rate limits for different APIs and different models within them. For example, consider the following json file `max_queries.json`:
```json
{
    "gemini": {
        "gemini-1.5-pro": 20
    },
    "openai": {
        "gpt4": 10,
        "gpt3.5-turbo": 20
    }
}
```

Note that the rate limit for the "gemini-1.0-pro" model is not defined in the json file as well as the "ollama" API. This means that the rate limit for these model will be set to the default rate limit which is set using the `--max-queries` flag.

In general, _you only specify the rate limits for the models that you want to set a different rate limit for_ - everything that is not specified will be set to the default rate limit.

Then we can run the experiment with the following command:
```bash
prompto_run_experiment \
    --file path/to/experiment.jsonl \
    --data-folder data \
    --max-queries 5 \
    --max-queries-json max_queries.json \
    --parallel
```

In this case, there are actually 6 groups/queues of prompts created (although not all of them will have prompts in the queues):
1. `gemini-gemini-1.0-pro`: Gemini API with model "gemini-1.0-pro" with rate limit of 20
2. `gemini`: Gemini API with rate limit of 5 (default rate limit provided) - i.e. all the prompts with the "gemini" API that are not "gemini-1.5-pro"
3. `openai-gpt4`: OpenAI API with model "gpt4" with rate limit of 10
4. `openai-gpt3.5-turbo`: OpenAI API with model "gpt3.5-turbo" with rate limit of 20
5. `openai`: OpenAI API with rate limit of 5 (default rate limit provided) - i.e. all the prompts with the "openai" API that are not "gpt4" or "gpt3.5-turbo"
6. `ollama`: Ollama API with rate limit of 5 (default rate limit provided) - i.e. all the prompts with the "ollama" API

Note here that:
- Group 5 (`openai`) here does not have any prompts in it as all the prompts with the "openai" API are either "gpt4" or "gpt3.5-turbo"
- Groups 2 (`gemini`), 5 (`openai`) and 6 (`ollama`) are generated by the API types which will always be generated if the `--parallel` flag is set
- Groups 1 (`gemini-gemini-1.0-pro`), 3 (`openai-gpt4`) and 4 (`openai-gpt3.5-turbo`) are generated by the models which are generated by the keys in the sub-dictionaries of the `max_queries_dict`

If we wanted to adjust the default rate limit for a given API type, we can do so by specifing a rate limit for `"default"` in the sub-dictionary. For example, consider the following json file `max_queries.json`:
```json
{
    "gemini": {
        "default": 30,
        "gemini-1.5-pro": 20
    },
    "openai": {
        "gpt4": 10,
        "gpt3.5-turbo": 20
    },
    "ollama": 4
}
```

In this case, the rate limit for the "ollama" API is set to 4 queries per minute - this is done just like how we set rate limits for each API in the [above section](#different-rate-limits-for-each-api-type). The change here is that for Group 2 (the group/queue for the "gemini" API which are not for the "gemini-1.5-pro" model), the rate limit is set to 30 queries per minute.

Note for specifying the "ollama" API, writing `"ollama": 4` is equivalent to writing `"ollama": {"default": 4}`.

Again it is important to note that the keys in the json file must match the values of the "api" and "model_name" keys in the experiment file. If there is something misspelled in the experiment file, then the rate limit for that API or model will be set to the default rate limit which is set using the `--max-queries` flag.

### Full control: Using the "groups" key to define user-specified groups of prompts

If you want full control over how the prompts are split into different groups/queues, you can use the "groups" key in the experiment file to define user-specified groups of prompts. This is useful when you want to group the prompts in a way that is not based on the "api" key. For example, consider the following experiment file:
```json
{"id": 0, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of France?", "group": "group1"}
{"id": 1, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of Germany?", "group": "group2"}
{"id": 2, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of France?", "group": "group1"}
{"id": 3, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of Germany?", "group": "group2"}
{"id": 4, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of France?", "group": "group1"}
{"id": 5, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of Germany?", "group": "group2"}
{"id": 6, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of France?", "group": "group1"}
{"id": 7, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of Germany?", "group": "group2"}
{"id": 8, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of France?", "group": "group3"}
{"id": 9, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of Germany?", "group": "group3"}
{"id": 10, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of France?", "group": "group3"}
{"id": 11, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of Germany?", "group": "group3"}
```

In this case, we have defined 3 groups of prompts: "group1", "group2" and "group3". We can then set the rate limits for each of these groups using the `--max-queries-json` flag. For example, consider the following json file `max_queries.json`:
```json
{
    "group1": 5,
    "group2": 10,
    "group3": 15
}
```

#### Mixing using the "api" and "group" keys to define groups

It is possible to have an experiment file where only some of the prompts have a "group" key. This can be useful in cases where you might want to only group a few prompts within a certain API type. An example might be if one had two Ollama endpoints and wanted to split up the prompts to different models to the different Ollama endpoints they had available to them. For example, consider the following experiment file:
```json
{"id": 0, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of France?"}
{"id": 1, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "What is the capital of Germany?"}
{"id": 2, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of France?"}
{"id": 3, "api": "gemini", "model_name": "gemini-1.5-pro", "prompt": "What is the capital of Germany?"}
{"id": 4, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of France?"}
{"id": 5, "api": "openai", "model_name": "gpt3.5-turbo", "prompt": "What is the capital of Germany?"}
{"id": 6, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of France?"}
{"id": 7, "api": "openai", "model_name": "gpt4", "prompt": "What is the capital of Germany?"}
{"id": 8, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of France?", "group": "group1"}
{"id": 9, "api": "ollama", "model_name": "llama3", "prompt": "What is the capital of Germany?", "group": "group1"}
{"id": 10, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of France?", "group": "group1"}
{"id": 11, "api": "ollama", "model_name": "mistral", "prompt": "What is the capital of Germany?", "group": "group1"}
{"id": 12, "api": "ollama", "model_name": "gemma", "prompt": "What is the capital of France?", "group": "group2"}
{"id": 13, "api": "ollama", "model_name": "gemma", "prompt": "What is the capital of Germany?", "group": "group2"}
{"id": 14, "api": "ollama", "model_name": "phi3", "prompt": "What is the capital of France?", "group": "group2"}
{"id": 15, "api": "ollama", "model_name": "phi3", "prompt": "What is the capital of Germany?", "group": "group2"}
```

In this case, we have defined 2 groups of prompts: "group1" and "group2". We can then set the rate limits for each of these groups using the `--max-queries-json` flag. For example, consider the following json file `max_queries.json`:
```json
{
    "group1": 5,
    "group2": 10
}
```

We can then run the experiment with the following command:
```bash
prompto_run_experiment --file path/to/experiment.jsonl --data-folder data --max-queries 5 --max-queries-json max_queries.json --parallel
```

In this case, we are creating two queues which have "ollama" prompts. One of these are for "llama3" and "mistral" models and the other is for "gemma" and "phi3" models. The rate limit of 5 queries per minute is applied to the "group1" queue and the rate limit of 10 queries per minute is applied to the "group2" queue.

In addition, we also have the separate queues for each API type which are generated by the API types which will always be generated if the `--parallel` flag is set.

In this example, a total of 4 queues are created:
1. `gemini`: Gemini API with rate limit of 5
2. `openai`: OpenAI API with rate limit of 5
3. `group1`: Ollama API with "llama3" and "mistral" models with rate limit of 5
4. `group2`: Ollama API with "gemma" and "phi3" models with rate limit of 10
