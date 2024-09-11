# Evaluation

A common use case for `prompto` is to evaluate the performance of different models on a given task where we first need to obtain a large number of responses.
In `prompto`, we provide functionality to automate the querying of different models and endpoints to obtain responses to a set of prompts and _then evaluate_ these responses.

## Automatic evaluation using an LLM-as-judge

To perform an LLM-as-judge evaluation, we essentially treat this as just _another_ `prompto` experiment where we have a set of prompts (which are now some judge evaluation template including the response from a model) and we query another model to obtain a judge evaluation response.

Therefore, given a _completed_ experiment file (i.e., a jsonl file where each line is a json object containing the prompt and response from a model), we can create another experiment file where the prompts are generated using some judge evaluation template and the completed response file. We must specify the model that we want to use as the judge. We call this a _judge_ experiment file and we can use `prompto` again to run this experiment and obtain the judge evaluation responses.

### Judge folder

To run an LLM-as-judge evaluation, you must first create a _judge folder_ consisting of:
```
└── judge_folder
    └── settings.json: a dictionary where keys are judge identifiers
        and the values are also dictionaries containing the "api",
        "model_name", and "parameters" to specify the LLM to use as a judge.
    ...
    └── template .txt files: several template files that specify how to
        generate the prompts for the judge evaluation
```

#### Judge settings file

For instance, the `settings.json` file could look like this:
```json
{
    "gemini-1.0-pro": {
        "api": "gemini",
        "model_name": "gemini-1.0-pro",
        "parameters": {"temperature": 0.5}
    },
    "gpt-4": {
        "api": "openai",
        "model_name": "gpt-4",
        "parameters": {"temperature": 0.5}
    }
}
```

We will see later that the commands for creating or running a judge evaluation will require the `judge` argument where we specify the judge identifier given by the keys of the `settings.json` file (e.g., `gemini-1.0-pro` or `gpt-4` in this case).

#### Template files

For creating a judge experiment, you must provide a prompt template which will be used to generate the prompts for the judge evaluation. This template should contain the response from the model that you want to evaluate. For instance, a basic template might look something like:
```
Given the following input and output of a model, please rate the quality of the response:
Input: {INPUT_PROMPT}
Response: {OUTPUT_RESPONSE}
```

We allow for specifying multiple templates (for different evaluation prompts), so you might have several `.txt` files in the judge folder, so you might have a folder looking like:
```
└── judge_folder
    └── settings.json
    └── template.txt
    └── template2.txt
    ...
```

We will see later that the commands for creating or running a judge evaluation has a `templates` argument where you can specify a comma-separated list of template files (e.g., `template.txt,template2.txt`). By default, this is `template.txt` if not specified.

### Using `prompto` for LLM-as-judge evaluation

`prompto` also allows you to run a LLM-as-judge evaluation when running the experiment the first time (using [`prompto_run_experiment`](./commands.md#running-an-experiment-file)) by doing this in a two step process:
1. Run the original `prompto` experiment with the models you want to evaluate and save the responses to a file
2. Create a judge experiment file using the responses from the first experiment and run the judge experiment

We will first show how to create a judge experiment file (given an already completed experiment), and then show the how to run the judge experiment directly when using `prompto_run_experiment`.

### Creating a judge experiment file from a completed experiment

Given a completed experiment file, we can create a judge experiment file using the [`prompto_create_judge_file` command](./commands.md#create-judge-file). To see all arguments of this command, run `prompto_create_judge_file --help`.

To create a judge experiment file for a particular experiment file with a judge-folder as `./judge`, we can use the following command:
```
prompto_create_judge_file \
    --experiment-file path/to/experiment.jsonl \
    --judge-folder judge \
    --templates template.txt \
    --judge gemini-1.0-pro
```

This would generate a new experiment file with prompts generated using the template in `judge/template.txt` and the responses from the completed experiment file. The `--judge` argument specifies the judge identifier to use from the `judge/settings.json` file in the judge folder, so in this case, it would use the `gemini-1.0-pro` model as the judge - this specifies the `api`, `model_name`, and `parameters` to use for the judge LLM.

As noted above, it's possible to use multiple templates and multiple judges by specifying a comma-separated list of template files and judge identifiers, for instance:
```
prompto_create_judge_file \
    --experiment-file path/to/experiment.jsonl \
    --judge-folder judge \
    --templates template.txt,template2.txt \
    --judge gemini-1.0-pro,gpt-4
```

Here, for each prompt dictionary in the completed experiment file, there would be 4 prompts generated (from the 2 templates and 2 judges). The full number of prompts generated would be `num_templates * num_judges * num_prompts_in_experiment_file`.

This will create a new experiment file

### Running a LLM-as-judge evaluation automatically using `prompto_run_experiment`

It is also possible to run a LLM-as-judge evaluation directly when running the experiment the first time using the [`prompto_run_experiment`](./commands.md#running-an-experiment-file) command. To do this, you just use the same arguments as described above. For instance, to run an experiment file with automatic evaluation using a judge, you can use the following command:
```
prompto_run_experiment \
    --file path/to/experiment.jsonl \
    --data-folder data \
    --judge-folder judge \
    --templates template.txt,template2.txt \
    --judge gemini-1.0-pro
```

This command would first run the experiment file to obtain responses for each prompt, then create a new judge experiment file using the completed responses and the templates in `judge/template.txt` and `judge/template2.txt`, and lastly run the judge experiment using the `gemini-1.0-pro` model specified in the `judge/settings.json` file.

## Automatic evaluation using a scoring function

`prompto` supports automatic evaluation using a scoring function. A scoring function is typically something which is lightweight such as performing string matching or regex computation. For `prompto` a scoring function is defined as any function that takes in a completed prompt dictionary and returns a dictionary with new keys that define some score for the prompt.

For example, we have some built-in scoring functions in [src/prompto/scorers.py](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/scorer.py):
- `match()`: takes in a completed prompt dictionary `prompt_dict` as an argument and sets a new key "match" which is `True` if `prompt_dict["response"`]==`prompt_dict["expected_response"]` and `False` otherwise.
- `includes()`: takes in a completed prompt dictionary `prompt_dict` as an argument and sets a new key "includes" which is `True` if `prompt_dict["response"`] includes `prompt_dict["expected_response"]` and `False` otherwise.

It is possible to define your own scoring functions by creating a new function in a Python file. The only restriction is that it must take in a completed prompt dictionary as an argument and return a dictionary with new keys that define some score for the prompt, i.e. it has the following structure:
```python
def my_scorer(prompt_dict: dict) -> dict:
    # some computation to score the response
    prompt_dict["my_score"] = <something>
    return prompt_dict
```

### Using a scorer in `prompto`

In Python, to use a scorer, when processing an experiment, you can pass in a list of scoring functions to the `Experiment.process()` method. For instance, you can use the `match` and `includes` scorers as follows:
```python
from prompto.scorers import match, includes
from prompto.settings import Settings
from prompto.experiment import Experiment

settings = Settings(data_folder="data")
experiment = Experiment(file_name="experiment.jsonl", settings=settings)
experiment.process(evaluation_funcs=[match, includes])
```

Here, you could also include any other custom functions in the list passed for `evaluation_funcs`.

For a more detailed notebook walkthrough, see the [Running experiments with custom evaluations notebook](https://github.com/alan-turing-institute/prompto/blob/main/examples/evaluation/Running_experiments_with_custom_evaluations.ipynb)

### Running a scorer evaluation automatically using `prompto_run_experiment`

In the command line, you can use the `--scorers` argument to specify a list of scoring functions to use. To do so, you must first add the scoring function to the `SCORING_FUNCTIONS` dictionary in [src/prompto/scorers.py](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/scorer.py) (this is at the bottom of the file). You can then pass in the key corresponding to the scoring function to the `--scorers` argument as a comma-separated list. For instance, to run an experiment file with automatic evaluation using the `match` and `includes` scorers, you can use the following command:
```
prompto_run_experiment \
    --file path/to/experiment.jsonl \
    --data-folder data \
    --scorers match,includes
```

This will run the experiment file and for each prompt dictionary, the `match` and `includes` scoring functions will be applied to the completed prompt dictionary (and the new "match" and "includes" keys will be added to the prompt dictionary).

For custom scoring functions, you must do the following:
1. Implement the scoring function in either a Python file or in the [src/prompto/scorers.py](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/scorer.py) file (if it's in another file, you'll just need to import it in the `src/prompto/scorers.py` file)
2. Add it to the `SCORING_FUNCTIONS` dictionary in the [src/prompto/scorers.py](https://github.com/alan-turing-institute/prompto/blob/main/src/prompto/scorer.py) file
3. Pass in the key corresponding to the scoring function to the `--scorers` argument
