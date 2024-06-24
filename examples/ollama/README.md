# Using `prompto` with Ollama

For prompts to Ollama API, you can simply add a line in your experiment (`.jsonl`) file where you specify the `api` to be `ollama`. See the [models doc](../../docs/models.md#ollama) for some details on the environment variables you need to set.

By default, the address and port that Ollama uses when running is `localhost:11434` and so when running Ollama locally, we set the `OLLAMA_API_ENDPOINT` to `http://localhost:11434"`. If you are running the server at a different address or port, you can specify with the `OLLAMA_API_ENDPOINT` environment variable. See the [Setting up Ollama locally](./ollama.ipynb#setting-up-ollama-locally) section in the [ollama.ipynb](./ollama.ipynb) notebook for more details.

We provide an example experiment file in [./data/input/ollama-example.jsonl](./data/input/ollama-example.jsonl). You can run it with the following command (assuming that your working directory is the current directory of this notebook, i.e. `examples/ollama`):
```bash
prompto_run_experiment --file data/input/ollama-example.jsonl --max_queries 30
```

To run the experiment, you will need to set the following environment variables first:
```bash
export OLLAMA_API_ENDPOINT=<YOUR-OLLAMA-ENDPOINT> # if running locally, set to http://localhost:11434
export OLLAMA_MODEL_NAME=<DEFAULT-OLLAMA-MODEL>
```

You can also use an `.env` file to save these environment variables without needing to export them globally in the terminal:
```
OLLAMA_API_ENDPOINT=<YOUR-OLLAMA-ENDPOINT>
OLLAMA_MODEL_NAME=<DEFAULT-OLLAMA-MODEL>
```

By default, the `prompto_run_experiment` command will look for an `.env` file in the current directory. If you want to use a different `.env` file, you can specify it with the `--env` flag.

Also see the [ollama.ipynb](./ollama.ipynb) notebook for a more detailed walkthrough on the how to set the environment variables and run the experiment and the different types of prompts you can run.

Do note that when you run the experiment, the input file ([./data/input/ollama-example.jsonl](./data/input/ollama-example.jsonl)) will be moved to the output directory (timestamped for when you run the experiment).
