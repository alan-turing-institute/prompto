# Using `prompto` with OpenAI

For prompts to OpenAI API, you can simply add a line in your experiment (`.jsonl`) file where you specify the `api` to be `openai`. See the [models doc](../../docs/models.md#openai) for some details on the environment variables you need to set.

We provide an example experiment file in [data/input/openai-example.jsonl](./data/input/openai-example.jsonl). You can run it with the following command (assuming that your working directory is the current directory of this notebook, i.e. `examples/openai`):
```bash
prompto_run_experiment --file data/input/openai-example.jsonl --max_queries 30
```

To run the experiment, you will need to set the following environment variables first:
```bash
export OPENAI_API_KEY=<YOUR-OPENAI-KEY>
```

You can also use an `.env` file to save these environment variables without needing to export them globally in the terminal:
```
OPENAI_API_KEY=<YOUR-OPENAI-KEY>
```

By default, the `prompto_run_experiment` command will look for an `.env` file in the current directory. If you want to use a different `.env` file, you can specify it with the `--env` flag.

Also see the [openai.ipynb](./openai.ipynb) notebook for a more detailed walkthrough on the how to set the environment variables and run the experiment and the different types of prompts you can run.

Do note that when you run the experiment, the input file ([data/input/openai-example.jsonl](./data/input/openai-example.jsonl)) will be moved to the output directory (timestamped for when you run the experiment).
