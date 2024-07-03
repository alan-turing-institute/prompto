# Instructions to add new API/model

The `prompto` library supports querying multiple LLM API endpoints asynchronously (see [available APIs](./../README.md#available-apis-and-models) and the [model docs](./models.md)). However, the list of available APIs is far from complete! As we don't have access to every API available, we need your help to implement them and welcome contributions to the library! It might also be the case that an API has been implemented, but perhaps it needs to updated or improved.

In this document, we aim to capture some key steps to add a new API/model to the library. We hope that this will develop into a helpful guide.

For a guide to contributing to the library in general, see our [contribution guide](./contribution.md). If you have any suggestions or corrections, please feel free to contribute!

## The `prompto` library structure

The source code for the library can be found in the [`src/prompto`](https://github.com/alan-turing-institute/prompto/tree/main/src/prompto) directory of the repository. The main components of the library are:

* [`src/prompto/settings.py`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/settings.py): this includes the definition of the `Settings` class which defines the settings to use when running experiments or the pipeline and stores the paths to the relevant data folders and the parameters for the pipeline.
* [`src/prompto/experiment.py`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/experiment.py): this includes the definition of an experiment and stores all the variables related to a _single_ experiment. An 'experiment' here is defined by a particular JSONL file which contains the data/prompts for each experiment. Each line in this file is a particular input to the LLM which we will obtain a response for
    * An experiment can be processed by calling the `Experiment.process()` method which will query the model and store the results in the output folder.
    * This is what is called when running the [`prompto_run_experiment`](./commands.md#running-an-experiment-file) command.
* [`src/prompto/experiment_pipeline.py`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/experiment_pipeline.py): this is the main class for running the full pipeline. The pipeline can be ran using the `ExperimentPipeline.run()` method which will continually check the input folder for new experiments to process.
    * This is what is called when running the [`prompto_run_pipeline`](./commands.md#running-the-pipeline) command.

You will notice in this source code, there is also a [`src/prompto/apis`](https://github.com/alan-turing-institute/prompto/tree/main/src/prompto/apis) directory which contains further sub-directories for implementations of the different APIs. For adding a new API, you will need to create a new sub-directory in this `apis` directory. Within this sub-directory, you will need to create a new Python file which will contain the implementation of the API.

You can add a new API by creating a new class which inherits from the `AsyncAPI` class in the [`src/prompto/apis/base.py`](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/apis/base.py). This class defines the basic structure of an API and includes some key methods that need to be implemented for querying the model asynchronously.

```python
from typing import Any

from prompto.apis.base import AsyncAPI
from prompto.settings import Settings

class MyNewAPI(AsyncAPI):
    def __init__(
        self,
        settings: Settings,
        log_file: str,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(settings=settings, log_file=log_file, *args, **kwargs)
        # Add any additional initialisation code here

    @staticmethod
    def check_environment_variables() -> list[Exception]:
        # Implement the environment variable checks here
        pass

    @staticmethod
    def check_prompt_dict(prompt_dict: dict) -> list[Exception]:
        # Implement the prompt_dict checks here
        pass

    async def query(
        self,
        prompt_dict: dict,
        index: int | str = "NA",
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        # Implement the async querying logic here to "complete" a prompt_dict
        pass
```

The critical method to implement is the `query` method which should take a `prompt_dict` (a dictionary containing the prompt and any other parameters - this is directly a line in the input jsonl file) and returns a "completed" `prompt_dict` which includes a `"response"` key with the response from the model. The `query` method should be asynchronous.

The `check_environment_variables` and `check_prompt_dict` methods are only used for running prior checks on an experiment file. These checks are split up into two kinds:

* `check_environment_variables`: checks that the environment variables required for the API are set
* `check_prompt_dict`: checks that the `prompt_dict` is correctly formatted and contains all the required keys for querying the API

These checks are run when using the `prompto_check_experiment` command. These methods are recommended to be implemented to ensure that the API is correctly set up, but are not necessary for the API to function and for `prompto` to query the model.
