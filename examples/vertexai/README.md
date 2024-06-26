# Using `prompto` with Vertex AI

For prompts to Vertex AI API, you can simply add a line in your experiment (`.jsonl`) file where you specify the `api` to be `vertexai`. See the [models doc](../../docs/models.md#vertex-ai) for some details on the environment variables you need to set.

Note that the Vertex AI API is different to the Gemini API. For Gemini API, see the [gemini](../gemini) example.

We provide an example experiment file in [data/input/vertexai-example.jsonl](./data/input/vertexai-example.jsonl). You can run it with the following command (assuming that your working directory is the current directory of this notebook, i.e. `examples/vertexai`):
```bash
prompto_run_experiment --file data/input/vertexai-example.jsonl --max_queries 30
```

To run the experiment, you will need to set the following environment variables first:
```bash
export VERTEXAI_PROJECT_ID=<YOUR-VERTEXAI-PROJECT-ID>
export VERTEXAI_LOCATION_ID=<YOUR-VERTEXAI-LOCATION-ID>
```

When using Vertex AI, you need to set up the [gcloud CLI](https://cloud.google.com/cli) and authenticate with your Google Cloud account. In particular, you will have ran
```bash
gcloud auth application-default login
```

After this, you can set your default project ID with
```bash
gcloud config set project <YOUR-VERTEXAI-PROJECT-ID>
```

By doing this, you can optionally choose to not set the `VERTEXAI_PROJECT_ID` environment variable and the `vertexai` library will use the default project ID set by the `gcloud` CLI.

You can also use an `.env` file to save these environment variables without needing to export them globally in the terminal:
```
VERTEXAI_PROJECT_ID=<YOUR-VERTEXAI-PROJECT-ID>
VERTEXAI_LOCATION_ID=<YOUR-VERTEXAI-LOCATION-ID>
```

By default, the `prompto_run_experiment` command will look for an `.env` file in the current directory. If you want to use a different `.env` file, you can specify it with the `--env` flag.

Also see the [vertexai.ipynb](./vertexai.ipynb) notebook for a more detailed walkthrough on the how to set the environment variables and run the experiment and the different types of prompts you can run.

Do note that when you run the experiment, the input file ([data/input/vertexai-example.jsonl](./data/input/vertexai-example.jsonl)) will be moved to the output directory (timestamped for when you run the experiment).
