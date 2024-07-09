# Environment variables

Each API has a number of environment variables that are either required or optional to be set in order to query the model. We recommend setting these environment variables in a `.env` file in the root of the project directory. When you run the `prompto_run_experiment` or `prompto_run_pipeline` commands, the library will look for an `.env` file in the current directory and we then use `python-dotenv` to load the environment variables into the Python environment.

An `.env` file would look something like:
```
OPENAI_API_KEY=<YOUR-OPENAI-KEY>
AZURE_OPENAI_API_KEY=<YOUR-AZURE-OPENAI-KEY>
OLLAMA_API_ENDPOINT=<YOUR-OLLAMA_API-ENDPOINT>
```

This allows you to not necessarily need to load them into the global environment in your terminal each time you run an experiment.

Alternatively, you can set the environment variables in your terminal using `export ENVIRONMENT_VARIABLE=value` or having another bash script to set these without a `.env` file * this file would look similar to the example `.env` file, but with `export` in front of each line so that it is loaded into the global environment.

We list the environment variables for each API in the their respective documentation pages. See the [models doc](./models.md) for a list of all the models and their respective documentation pages.

## Model-specific environment variables

There may be some experiments where you wish to query different models from the same API type (e.g. querying different deployments of models from an Azure OpenAI API). In such cases, you can specify the model name in the `model_name` key of the `prompt_dict`.

There may be some cases where the environment variables needed for different models are different, for example if you are querying models which have different API keys (in Azure OpenAI, you may have different models that live on different subscriptions). In such cases, you can set _model-specific environment variables_ by appending the model name to the environment variable name (with a underscore between the environment variable and the model name): `ENVIRONMENT_VARIABLE -> ENVIRONMENT_VARIABLE_model_name`. Note that the model name might have to be formatted to be a valid environment variable name (e.g. replacing `("-", "/", ".", " ")` with underscores `_`).

For example, if you have two models in Azure OpenAI, `model1` and `model2`, you can set the environment variables `AZURE_OPENAI_API_KEY_model1` and `AZURE_OPENAI_API_KEY_model2` with the respective API keys for each model.

The base environment variable name (e.g. `AZURE_OPENAI_API_KEY`) is used as a _default environment variable_ if the model-specific environment variable is not set. They are used either in the case where the model name is not specified in the `prompt_dict` or if the model-specific environment variable is not set.

To clarify, the order of precedence for the API key is as follows:

1. If `model_name` is specified in the `prompt_dict`, the model-specific environment variable is used
2. If `model_name` is specified in the `prompt_dict` but the model-specific environment variable is not set, the default environment variable is used
3. If `model_name` is not specified in the `prompt_dict`, the default environment variable is used
4. If neither the model-specific environment variable nor the default environment variable is set, an error is raised
