# prompto

`prompto` derives from the Italian word "_pronto_" which means "_ready_" and could also mean "_I prompt_" in Italian (if "_promptare_" was a verb meaning "_to prompt_").

`prompto` is a Python library facilitates of LLM experiments stored as jsonl files. It automates querying API endpoints and logs progress asynchronously. The library is designed to be extensible and can be used to query different models.

## Getting Started

The library has functionality to process experiments and to run a pipeline which continually looks for new experiment jsonl files in the input folder. Everything starts with defining a **pipeline data folder** which contains:
- `input` folder: contains the jsonl files with the experiments
- `output` folder: where the results of the experiments will be stored. When an experiment is ran, a folder is created within the output folder of the experiment name (as defined in the jsonl file but removing the `.jsonl` extension) and the results and logs for the experiment are stored there
- `media` folder: which contains the media files for the experiments. These files must be within folders of the same experiment name (as defined in the jsonl file but removing the `.jsonl` extension)

When using the library, you simply pass in the folder you would like to use as the pipeline data folder and the library will take care of the rest.

The main command line interface for running an experiment is the `prompto_run_experiment` command (see the [commands doc](docs/commands.md) for more details). This command will process a single experiment file and query the model for each prompt in the file. The results will be stored in the output folder of the experiment. To see all arguments of this command, run:
```bash
prompto_run_experiment --help
```

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
2. Create a folder in the the `output` folder with the name of the experiment (the file name without the `.jsonl` extention - in this case, `openai`)
3. Move the `openai.jsonl` file to the `output/openai` folder (and add a timestamp of when the input file was created to that file)
4. Start running the experiment and sending requests to the OpenAI API asynchronously which we specified in this command to be 30 queries a minute (so requests are sent every 2 seconds) - the default is 10 queries per minute
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
│   │       └── DD-MM-YYYY-hh-mm-ss-openai-log.txt
├── .env
```

The completed experiment file will contain the responses from the OpenAI API for the specific model in each prompt in the input file in `data/output/openai/DD-MM-YYYY-hh-mm-ss-completed-openai.jsonl` where `DD-MM-YYYY-hh-mm-ss` is the timestamp of when the input file was created.

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
│   │       └── DD-MM-YYYY-hh-mm-ss-gemini-log.txt
├── .env
```

The completed experiment file will contain the responses from the Gemini API for the specified model in each prompt in the input file in `data/output/gemini/DD-MM-YYYY-hh-mm-ss-completed-gemini.jsonl` where `DD-MM-YYYY-hh-mm-ss` is the timestamp of when the input file was created.

## Using the Library in Python

The library has a few key classes:
- [`Settings`](src/prompto/settings.py): this defines the settings of the the experiment pipeline which stores the paths to the relevant data folders and the parameters for the pipeline.
- [`Experiment`](src/prompto/experiment.py): this defines all the variables related to a _single_ experiment. An 'experiment' here is defined by a particular JSONL file which contains the data/prompts for each experiment. Each line in this folder is a particular input to the LLM which we will obtain a response for. An experiment can be processed by calling the `Experiment.process()` method which will query the model and store the results in the output folder.
- [`ExperimentPipeline`](src/prompto/experiment_pipeline.py): this is the main class for running the full pipeline. The pipeline can be ran using the `ExperimentPipeline.run()` method which will continually check the input folder for new experiments to process.
- [`AsyncBaseAPI`](src/prompto/base.py): this is the base class for querying all APIs. Each API/model should inherit from this class and implement the `async_query` method which will (asynchronously) query the model's API and return the response. When running an experiment, the `Experiment` class will call this method for each experiment to send requests asynchronously.

When a new model is added, you must add it to the [`API`](src/prompto/apis/__init__.py) dictionary which is in the `apis` module. This dictionary should map the model name to the class of the model.

### Prerequisites

Before running the script, ensure you have the following:

- Python >= 3.11
- Poetry (for dependency management)

### Installation

1. **Clone the Repository**
    ```bash
    git clone git@github.com:alan-turing-institute/prompto.git
    ```

2. **Navigate to Project Directory**
    ```bash
    cd prompto
    ```

3. **Install Poetry**
    If you haven't installed Poetry yet, you can install it by following the instructions [here](https://python-poetry.org/docs/#installation).

4. **Create and activate a Poetry Environment**
    ```bash
    poetry shell
    ```

    **Note**: You can also use another virtual environment manager, such as `venv` or `conda` for this step if you prefer.

5. **Install Dependencies**
    ```bash
    poetry install
    ```

**Note**: This only installs the base dependencies. There are also extra group dependencies depending on the models that you'd like to query. For example, if you'd like to query models from the Gemini API, you can install the extra dependencies by running:
```bash
poetry install -E gemini
```
