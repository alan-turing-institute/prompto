# Using `prompto` with Anthropic

For prompts to Azure OpenAI API, you can simply add a line in your experiment (`.jsonl`) file where you specify the `api` to be `anthropic`. See the [models doc](./../../docs/anthropic.md) for some details on the environment variables you need to set.

We provide an example experiment file in [data/input/anthropic-example.jsonl](https://github.com/alan-turing-institute/prompto/blob/main/examples/anthropic/data/input/anthropic-example.jsonl). You can run it with the following command (assuming that your working directory is the current directory of this notebook, i.e. `examples/anthropic`):
```bash
prompto_run_experiment --file data/input/anthropic-example.jsonl --max-queries 30
```

To run the experiment, you will need to set the following environment variables first:
```bash
- `ANTHROPIC_API_KEY`: the API key for the Anthropic API
```

You can also use an `.env` file to save these environment variables without needing to export them globally in the terminal:
```
ANTHROPIC_API_KEY=<YOUR-ANTHROPIC-KEY>
```

By default, the `prompto_run_experiment` command will look for an `.env` file in the current directory. If you want to use a different `.env` file, you can specify it with the `--env` flag.

Also see the [anthropic.ipynb](./anthropic.ipynb) notebook for a more detailed walkthrough on the how to set the environment variables and run the experiment and the different types of prompts you can run.

Do note that when you run the experiment, the input file ([data/input/anthropic-example.jsonl](https://github.com/alan-turing-institute/prompto/blob/main/examples/anthropic/data/input/anthropic-example.jsonl)) will be moved to the output directory (timestamped for when you run the experiment).
