# prompto

`prompto` is a Python library which facilitates processing of experiments of Large Language Models (LLMs) stored as jsonl files. It automates _asynchronous querying of LLM API endpoints_ and logs progress.

`prompto` derives from the Italian word "_pronto_" which means "_ready_". It could also mean "_I prompt_" in Italian (if "_promptare_" was a verb meaning "_to prompt_").

## Why `prompto`?

The benefit of  _asynchronous querying_ is that it allows for multiple requests to be sent to an API _without_ having to wait for the LLM's response, which is particularly useful to fully utilise the rate limits of an API. This is especially useful when an experiment file contains a large number of prompts and/or has several models to query. [_Asynchronous programming_](https://docs.python.org/3/library/asyncio.html) is simply a way for programs to avoid getting stuck on long tasks (like waiting for an LLM response from an API) and instead keep running other things at the same time (to send other queries).

With `prompto`, you are able to define your experiments of LLMs in a jsonl file where each line contains the prompt and any parameters to be used for a query of a model from a specific API. The library will process the experiment file and query models and store results. You are also  able to query _multiple_ models from _different_ APIs in a single experiment file and `prompto` will take care of querying the models _asynchronously_ and in _parallel_.

The library is designed to be extensible and can be used to query different models.

For more details on the library, see the [documentation](./docs/README.md) where you can find information on [how to set up an experiment file](./docs/experiment_file.md), [how to run experiments](./docs/pipeline.md), [how to configure environment variables](./docs/environment_variables.md), [how to specify rate limits for APIs and to use parallel processing](./docs/rate_limits.md) and much more.

See below for [installation instructions](#installation) and [quickstarts for getting started](#getting-started) with `prompto`.

## Available APIs and Models

The library supports querying several APIs and models. The following APIs are currently supported are:

* [OpenAI](./docs/openai.md) (`"openai"`)
* [Azure OpenAI](./docs/azure_openai.md) (`"azure-openai"`)
* [Gemini](./docs/gemini.md) (`"gemini"`)
* [Anthropic](./docs/anthropic.md) (`"anthropic"`)
* [Vertex AI](./docs/vertexai.md) (`"vertexai"`)
* [Huggingface text-generation-inference](./docs/huggingface_tgi.md) (`"huggingface-tgi"`)
* [Ollama](./docs/ollama.md) (`"ollama"`)
* [A simple Quart API](./docs/quart.md) for running models from [`transformers`](https://github.com/huggingface/transformers) locally (`"quart"`)

Our aim for `prompto` is to support more APIs and models in the future and to make it easy to add new APIs and models to the library. We welcome contributions to add new APIs and models to the library. We have a [contribution guide](docs/contribution.md) and a [guide on how to add new APIs and models](./docs/add_new_api.md) to the library in the [docs](./docs/README.md).

## Installation

To install the library, you can use `pip`:
```bash
pip install prompto
```

**Note**: This only installs the base dependencies required for `prompto`. There are also extra group dependencies depending on the models that you'd like to query. For example, if you'd like to query models from the OpenAI and Gemini API, you can install the extra dependencies by running:
```bash
pip install prompto"[openai,gemini]"
```

To install all the dependencies for all the models, you can run:
```bash
pip install prompto[all]
```

You might also want to set up a development environment for the library. To do this, please refer to the [development environment setup guide](docs/contribution.md#setting-up-a-development-environment) in our [contribution guide](docs/contribution.md).

`prompto` derives from the Italian word "_pronto_" which means "_ready_" and could also mean "_I prompt_" in Italian (if "_promptare_" was a verb meaning "_to prompt_").

## Getting Started

The library has functionality to process experiments and to run a pipeline which continually looks for new experiment jsonl files in the input folder. Everything starts with defining a **pipeline data folder** which contains:
```
└── data
    └── input: contains the jsonl files with the experiments
    └── output: contains the results of the experiments runs.
        When an experiment is ran, a folder is created within the output folder with the experiment name
        as defined in the jsonl file but removing the `.jsonl` extension.
        The results and logs for the experiment are stored there
    └── media: contains the media files for the experiments.
        These files must be within folders of the same experiment name
        as defined in the jsonl file but removing the `.jsonl` extension
```

When using the library, you simply pass in the folder you would like to use as the pipeline data folder and the library will take care of the rest.

The main command line interface for running an experiment is the `prompto_run_experiment` command (see the [commands doc](docs/commands.md) for more details). This command will process a single experiment file and query the model for each prompt in the file. The results will be stored in the output folder of the experiment. To see all arguments of this command, run:
```bash
prompto_run_experiment --help
```

See the [examples](examples) folder for examples of how to use the library with different APIs/models. Each example contains an experiment file which contains prompts for the model(s) and a walkthrough on how to run the experiment.

### OpenAI example

The following is an example of an experiment file which contains two prompts for two different models from the OpenAI API:

```json
{"id": 0, "api": "openai", "model_name": "gpt-4o", "prompt": "How does technology impact us?", "parameters": {"n": 1, "temperature": 1, "max_tokens": 100}}
{"id": 1, "api": "openai", "model_name": "gpt-3.5-turbo", "prompt": "How does technology impact us?", "parameters": {"n": 1, "temperature": 1, "max_tokens": 100}}
```

To run this example, first install the library and create the following folder structure in your working directory from where you'll run this example:
```
├── data
│   └── input
│      └── openai.jsonl
├── .env
```
where `openai.jsonl` contains the above two prompts and the `.env` file contains the following:
```
OPENAI_API_KEY=<YOUR-OPENAI-KEY>
```

You are then ready to run the experiment with the following command:
```bash
prompto_run_experiment --file data/input/openai.jsonl --max-queries 30
```

This will:

1. Create subfolders in the `data` folder (in particular, it will create `media` (`data/media`) and `output` (`data/media`) folders)
2. Create a folder in the`output` folder with the name of the experiment (the file name without the `.jsonl` extention * in this case, `openai`)
3. Move the `openai.jsonl` file to the `output/openai` folder (and add a timestamp of when the run of the experiment started)
4. Start running the experiment and sending requests to the OpenAI API asynchronously which we specified in this command to be 30 queries a minute (so requests are sent every 2 seconds) * the default is 10 queries per minute
5. Results will be stored in a "completed" jsonl file in the output folder (which is also timestamped)
6. Logs will be printed out to the console and also stored in a log file (which is also timestamped)

The resulting folder structure will look like this:
```
├── data
│   ├── input
│   ├── media
│   ├── output
│   │   └── openai
│   │       ├── DD-MM-YYYY-hh-mm-ss-completed-openai.jsonl
│   │       ├── DD-MM-YYYY-hh-mm-ss-input-openai.jsonl
│   │       └── DD-MM-YYYY-hh-mm-ss-log-openai.txt
├── .env
```

The completed experiment file will contain the responses from the OpenAI API for the specific model in each prompt in the input file in `data/output/openai/DD-MM-YYYY-hh-mm-ss-completed-openai.jsonl` where `DD-MM-YYYY-hh-mm-ss` is the timestamp of when the experiment file started to be processed.

For a more detailed walkthrough on using `prompto` with the OpenAI API, see the [`openai` example](examples/openai).

### Gemini example

```json
{"id": 0, "api": "gemini", "model_name": "gemini-1.5-flash", "prompt": "How does technology impact us?", "safety_filter": "none", "parameters": {"candidate_count": 1, "temperature": 1, "max_output_tokens": 100}}
{"id": 1, "api": "gemini", "model_name": "gemini-1.0-pro", "prompt": "How does technology impact us?", "safety_filter": "few", "parameters": {"candidate_count": 1, "temperature": 1, "max_output_tokens": 100}}
```

To run this example, first install the library and create the following folder structure in your working directory from where you'll run this example:
```
├── data
│   └── input
│      └── gemini.jsonl
├── .env
```
where `gemini.jsonl` contains the above two prompts and the `.env` file contains the following:
```
GEMINI_API_KEY=<YOUR-GEMINI-KEY>
```

You are then ready to run the experiment with the following command:
```bash
prompto_run_experiment --file data/input/openai.jsonl --max-queries 30
```

As with the above example, the resulting folder structure will look like this:
```
├── data
│   ├── input
│   ├── media
│   ├── output
│   │   └── gemini
│   │       ├── DD-MM-YYYY-hh-mm-ss-completed-gemini.jsonl
│   │       ├── DD-MM-YYYY-hh-mm-ss-input-gemini.jsonl
│   │       └── DD-MM-YYYY-hh-mm-ss-log-gemini.txt
├── .env
```

The completed experiment file will contain the responses from the Gemini API for the specified model in each prompt in the input file in `data/output/gemini/DD-MM-YYYY-hh-mm-ss-completed-gemini.jsonl` where `DD-MM-YYYY-hh-mm-ss` is the timestamp of when the experiment file started to be processed.

For a more detailed walkthrough on using `prompto` with the Gemini API, see the [`gemini` example](examples/gemini).

## Using the Library in Python

The library has a few key classes:

* [`Settings`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/settings.py): this defines the settings of theexperiment pipeline which stores the paths to the relevant data folders and the parameters for the pipeline.
* [`Experiment`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/experiment.py): this defines all the variables related to a _single_ experiment. An 'experiment' here is defined by a particular JSONL file which contains the data/prompts for each experiment. Each line in this file is a particular input to the LLM which we will obtain a response for. An experiment can be processed by calling the `Experiment.process()` method which will query the model and store the results in the output folder.
* [`ExperimentPipeline`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/experiment_pipeline.py): this is the main class for running the full pipeline. The pipeline can be ran using the `ExperimentPipeline.run()` method which will continually check the input folder for new experiments to process.
* [`AsyncAPI`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/apis/base.py): this is the base class for querying all APIs. Each API/model should inherit from this class and implement the `query` method which will (asynchronously) query the model's API and return the response. When running an experiment, the `Experiment` class will call this method for each experiment to send requests asynchronously.

When a new model is added, you must add it to the [`API`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/apis/base.py) dictionary which is in the `apis` module. This dictionary should map the model name to the class of the model. For details on how to add a new model, see the [guide on adding new APIs and models](./docs/add_new_api.md).
