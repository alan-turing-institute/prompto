# Rephrasing prompts

It is often useful to be able to rephrase/paraphrase a given prompt, particularly in the area of evaluation of generative AI models. In `prompto`, we provide functionality to simply use another language model to to rephrase a given prompt. For this, we can start off by defining a set of prompts as usual (see [Setting up an experiment file](./experiment_file.md) documentation) and use the `Rephraser` class to create prompts to a model for rephrasal.

This _rephrasal experiment_ is simply just another set of prompts to a model where our prompts are now asking a model to rephrase/paraphrase a prompt/task. The responses to these prompts can then be used sent to another model for evaluation or for any other purpose.

Also see the [Rephrasing prompts with `prompto` notebook](../examples/evaluation/rephrase_prompts.ipynb) for a more detailed walkthrough the library for creating and running prompt rephrasal experiments.

### Rephrase folder

To run a rephrasal experiment, you must create a _rephrase folder_ consisting of:
```
└── judge_folder
    └── settings.json: a dictionary where keys are rephrasal model identifiers
        and the values are also dictionaries containing the "api",
        "model_name", and "parameters" to specify the LLM to use as a judge.
    └── template .txt: a txt file containing templates for the rephrasal prompts
```

#### Rephrase settings file

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

These define the models that we would like to use for rephrasal. When creating rephrasal examples, we pass in a list of rephrasal model identifiers so we know which models we want to use for rephrasal.

#### Template files

For creating a rephrasal experiment, we must provide a prompt template(s) which will be used to generate the prompts for rephrasing an original input prompt. The template should contain a placeholder for the original prompt `{INPUT_PROMPT}`. Each line in the file is read as a particular template. For example, a template file could look like this:
```
Write a paraphrase for the following sentence. Only reply with the paraphrased prompt. Prompt:\n"{INPUT_PROMPT}"
Write a variation of this sentence (only reply with the variation): "{INPUT_PROMPT}"
```

### Using `prompto` for rephrasal

`prompto` allows you to run a rephrasal experiment when running an experiment (using [`prompto_run_experiment`](./commands.md#running-an-experiment-file)). You can run a rephrasal experiment (experiment to obtain rephrased prompts) _and_ the rephrased experiment (the experiment using the rephrased prompts and optionally the original input prompts) in a single command by using:
```
prompto_run_experiment \
    --file path/to/experiment.jsonl \
    --data-folder data \
    --rephrase-folder rephrase \
    --rephrase-templates template.txt \
    --rephrase-model gemini-1.0-pro
```

This will:
1. run a rephrasal experiment using the prompts in `experiment.jsonl` experiment file using the "gemini-1.0-pro" model specified in the `settings.json` file in the `rephrase` folder
    - This experiment will generate rephrased prompts and outputs will be saved in a `rephrase-experiment` folder in the output folder (this is just the experiment name prefixed with `rephrase-`)
2. generate and run a new experiment file using the rephrased prompts and the original prompts in `experiment.jsonl`. The rephrased prompts are sent to the model specified in the `experiment.jsonl` file for the prompt it was rephrasing
    - The outputs will be saved in the output folder in `post-rephrased-experiment` folder (this is just the experiment name prefixed with `post-rephrased-`)

Note that you can also just run the rephrasal experiment (i.e. just run step 1 above) by using the `--only-rephrase` flag in the `prompto_run_experiment` command.

There is also a `--remove-original` flag which can be used to remove the original prompts from the new input file (and only have the rephrased prompts).
