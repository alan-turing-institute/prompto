# Commands

- [Running an experiment file](#running-an-experiment-file)
- [Running the pipeline](#running-the-pipeline)
- [Run checks on an experiment file](#run-checks-on-an-experiment-file)
- [Create judge file](#create-judge-file)
- [Obtain missing results jsonl file](#obtain-missing-results-jsonl-file)
- [Convert images to correct form](#convert-images-to-correct-form)
- [Start up Quart server](#start-up-quart-server)

## Running an experiment file

As detailed in the [pipeline documentation](pipeline.md), you can run a single experiment file using the `prompto_run_experiment` command and passing in a file. To see all arguments of this command, run `prompto_run_experiment --help`.

To run a particular experiment file with the data-folder set to the default path `./data`, you can use the following command:
```
prompto_run_experiment --file path/to/experiment.jsonl
```

This uses the default settings for the pipeline. You can also set the `--max-queries`, `--max-attempts`, and `--parallel` flags as detailed in the [pipeline documentation](pipeline.md).

If the experiment file is not in the input folder of the data folder, we will make a copy of the file in the input folder which will get processed. If you want to move the file to the input folder, you can use the `--move-to-input` flag:
```
prompto_run_experiment \
    --file path/to/experiment.jsonl \
    --data-folder data \
    --move-to-input
```

Note that if the experiment file is already in the input folder, we will not make a copy of the file and process the file in place.

## Running the pipeline

As detailed in the [pipeline documentation](pipeline.md), you can run the pipeline using the `prompto_run_pipeline` command. To see all arguments of this command, run `prompto_run_pipeline --help`.

To run a particular experiment file with the data-folder set to `pipeline-data`, you can use the following command:
```
prompto_run_pipeline --data-folder pipeline-data
```

This uses the default settings for the pipeline. You can also set the `--max-queries`, `--max-attempts`, and `--parallel` flags as detailed in the [pipeline documentation](pipeline.md).

## Run checks on an experiment file

It is possible to run a check over an experiment file to ensure that all the prompts are valid and the experiment file is correctly formatted. We also check for environment variables and log any errors or warnings that are found. To run this check, you can use the `prompto_check_experiment` command and passing in a file. To see all arguments of this command, run `prompto_check_experiment --help`.

To run a check on a particular experiment file, you can use the following command:
```
prompto_check_experiment --file path/to/experiment.jsonl
```

This will run the checks on the experiment file and log any errors or warnings that are found. You can optionally set the log-file to save the logs to a file using the `--log-file` flag (by default, it will be saved to a file in the current directory) and specify the path to the data folder using the `--data-folder` flag.

Lastly, it's possible to automatically move the file to the input folder of the data folder if it is not already there. To do this, you can use the `--move-to-input` flag:
```
prompto_check_experiment \
    --file path/to/experiment.jsonl \
    --data-folder data \
    --log-file path/to/logfile.txt \
    --move-to-input
```

## Create judge file

Once an experiment has been ran and responses to prompts have been obtained, it is possible to use another LLM as a "judge" to score the responses. This is useful for evaluating the quality of the responses obtained from the model. To create a judge file, you can use the `prompto_create_judge` command passing in the file containing the completed experiment and to a folder (i.e. judge location) containing the judge template and settings to use. To see all arguments of this command, run `prompto_create_judge --help`.

To create a judge file for a particular experiment file with a judge-location as `./judge` and using judge `gemini-1.0-pro` you can use the following command:
```
prompto_create_judge \
    --experiment-file path/to/experiment.jsonl \
    --judge-location judge \
    --judge gemini-1.0-pro
```

In `judge`, you must have two files:
- `template.txt`: this is the template file which contains the prompts and the responses to be scored. The responses should be replaced with the placeholders `{INPUT_PROMPT}` and `{OUTPUT_RESPONSE}`.
- `settings.json`: this is the settings json file which contains the settings for the judge(s). The keys are judge identifiers and the values are the "api", "model_name", "parameters" to specify the LLM to use as a judge (see the [experiment file documentation](experiment_file.md) for more details on these keys).

See for example [this judge example](../examples/data/data/judge) which contains example template and settings files.

The judge specified with the `--judge` flag should be a key in the `settings.json` file in the judge location. You can create different judge files using different LLMs as judge by specifying a different judge identifier from the keys in the `settings.json` file.

## Obtain missing results jsonl file

In some cases, you may have ran an experiment file and obtained responses for some prompts but not all. To obtain the missing results jsonl file, you can use the `prompto_obtain_missing_results` command passing in the input experiment file and the corresponding output experiment. You must also specify a path to a new jsonl file which will be created if any prompts are missing in the output file. The command looks at an ID key in the `prompt_dict`s of the input and output files to match the prompts, by default the name of this key is `id`. If the key is different, you can specify it using the `--id` flag. To see all arguments of this command, run `prompto_obtain_missing_results --help`.

To obtain the missing results jsonl file for a particular experiment file with the input experiment file as `path/to/experiment.jsonl`, the output experiment file as `path/to/experiment-output.jsonl`, and the new jsonl file as `path/to/missing-results.jsonl`, you can use the following command:
```
prompto_obtain_missing_results \
    --input-experiment path/to/experiment.jsonl \
    --output-experiment path/to/experiment-output.jsonl \
    --missing-results path/to/missing-results.jsonl
```

## Convert images to correct form

The `prompto_convert_images` command can be used to convert images to the correct form for the multimodal LLMs. This command takes in a folder containing images and checks if `.jpg`, `.jpeg` and `.png` files are saved in the correct format. If not, we resave them in the correct format.

To convert images in a folder `./images` to the correct form, you can use the following command:
```
prompto_convert_images --folder images
```

## Start up Quart server

As described in the [Quart API model documentation](models.md#quart-api), we have implemented a simple [Quart API](../src/prompto/apis/quart/quart_api.py) that can be used to quary a text-generation model from the [Huggingface model hub](https://huggingface.co/models) using the Huggingface `transformers` library. To start up the Quart server, you can use the `prompto_start_quart_server` command along with the Huggingface model name. To see all arguments of this command, run `prompto_start_quart_server --help`.

To start up the Quart server with [`vicgalle/gpt2-open-instruct-v1`](https://huggingface.co/vicgalle/gpt2-open-instruct-v1), at `"http://localhost:8000"`, you can use the following command:
```
prompto_start_quart_server \
    --model-name vicgalle/gpt2-open-instruct-v1 \
    --host localhost \
    --port 8000
```
