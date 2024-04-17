# batch-llm

This Python library facilitates batch processing of experiments stored as jsonl files. It automates querying the API and logs progress asynchronously. The library is designed to be extensible and can be used to query different models.

## Getting Started

(some key things to note - still need to finish)

The library has functionality to process experiments and to run a pipeline which continually looks for new experiment jsonl files in the input folder. Everything starts with defining a data folder which contains:
- `input` folder: contains the jsonl files with the experiments
- `output` folder: where the results of the experiments will be stored. When an experiment is ran, a folder is created within the output folder of the experiment name (as defined in the jsonl file but removing the `.jsonl` extension) and the results and logs for the experiment are stored there
- `media` folder: which contains the media files for the experiments. These files must be within folders of the same experiment name (as defined in the jsonl file but removing the `.jsonl` extension)

The library has a few key classes:
- [`Settings`](src/batch_llm/settings.py): this defines the settings of the the experiment pipeline which stores the paths to the relevant data folders and the parameters for the pipeline.
- [`Experiment`](src/batch_llm/experiment_processing.py): this defines all the variables related to a _single_ experiment. An 'experiment' here is defined by a particular JSONL file which contains the data/prompts for each experiment. Each line in this folder is a particular input to the LLM which we will obtain a response for.
- [`ExperimentPipeline`](src/batch_llm/experiment_processing.py): this is the main class for running the full pipeline. The pipeline can be ran using the `ExperimentPipeline.run()` method which will continually check the input folder for new experiments to process.
- [`BaseModel`](src/batch_llm/base.py): this is the base class for all models. Each model should inherit from this class and implement the `query` method which will query the model's API and return the response _and_ (more importantly) an `async_query` method. The `ExperimentPipeline` class will then call this method for each experiment to send requests asynchronously.

When a new model is added, you must add it to the [`MODELS`](src/batch_llm/models/__init__.py) dictionary which is in the `models` module. This dictionary should map the model name to the class of the model.

### Prerequisites

Before running the script, ensure you have the following:

- Python >= 3.11
- Poetry (for dependency management)

### Models

- Azure OpenAI
    - Need to set `OPENAI_API_KEY`, `AZURE_OPENAI_API_ENDPOINT` environment variables. You can also set the `AZURE_OPENAI_API_VERSION` variable too. Also recommended to set the `AZURE_OPENAI_MODEL_ID` in the environment variable to either avoid passing in the `model_name` each time if using the same one consistently.
- Gemini
    - Need to set `GEMINI_PROJECT_ID`, and `GEMINI_LOCATION` environment variables. Also recommended to set the `GEMINI_MODEL_ID` in the environment variable to either avoid passing in the `model_name` each time if using the same one consistently.

### Installation

1. **Clone the Repository**
    ```bash
    git clone git@github.com:alan-turing-institute/batch-llm.git
    ```

2. **Navigate to Project Directory**
    ```bash
    cd batch-llm
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
