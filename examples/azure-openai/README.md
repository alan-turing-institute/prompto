# Using `prompto` with AzureOpenAI

For prompts to Azure OpenAI API, you can simply add a line in your experiment (`.jsonl`) file where you specify the `api` to be `azure-openai`. See the [models doc](../../docs/models.md#azure-openai) for some details on the environment variables you need to set.

We provide an example experiment file in [data/input/azure-openai-example.jsonl](./data/input/azure-openai-example.jsonl). You can run it with the following command (assuming that your working directory is the current directory of this notebook, i.e. `examples/azure-openai`):
```bash
prompto_run_experiment --file data/input/azure-openai-example.jsonl --max_queries 30
```

To run the experiment, you will need to set the following environment variables first:
```bash
export AZURE_OPENAI_API_KEY=<YOUR-AZURE-OPENAI-KEY>
export AZURE_OPENAI_API_ENDPOINT=<YOUR-AZURE-OPENAI-ENDPOINT>
export AZURE_OPENAI_API_VERSION=<DEFAULT-AZURE-OPENAI-API-VERSION>
export AZURE_OPENAI_MODEL_NAME=<DEFAULT-AZURE-OPENAI-MODEL>
```

You can also use an `.env` file to save these environment variables without needing to export them globally in the terminal:
```
AZURE_OPENAI_API_KEY=<YOUR-AZURE-OPENAI-KEY>
AZURE_OPENAI_API_ENDPOINT=<YOUR-AZURE-OPENAI-ENDPOINT>
AZURE_OPENAI_API_VERSION=<DEFAULT-AZURE-OPENAI-API-VERSION>
AZURE_OPENAI_MODEL_NAME=<DEFAULT-AZURE-OPENAI-MODEL>
```

By default, the `prompto_run_experiment` command will look for an `.env` file in the current directory. If you want to use a different `.env` file, you can specify it with the `--env` flag.

Also see the [azure-openai.ipynb](./azure-openai.ipynb) notebook for a more detailed walkthrough on the how to set the environment variables and run the experiment and the different types of prompts you can run.

Do note that when you run the experiment, the input file ([data/input/azure-openai-example.jsonl](./data/input/azure-openai-example.jsonl)) will be moved to the output directory (timestamped for when you run the experiment).
