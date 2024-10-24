# Using `prompto` with Gemini

For prompts to Gemini API, you can simply add a line in your experiment (`.jsonl`) file where you specify the `api` to be `gemini`. See the [models doc](./../../docs/gemini.md) for some details on the environment variables you need to set.

Note that the Gemini API is different to the Vertex AI API. For Vertex AI API, see the [vertexai](./../vertexai/README.md) example.

We provide an example experiment file in [data/input/gemini-example.jsonl](https://github.com/alan-turing-institute/prompto/blob/main/examples/gemini/data/input/gemini-example.jsonl). You can run it with the following command (assuming that your working directory is the current directory of this notebook, i.e. `examples/gemini`):
```bash
prompto_run_experiment --file data/input/gemini-example.jsonl --max-queries 30
```

## Multimodal prompting

Multimodal prompting is available with the Gemini API. We provide an example notebook in the [Multimodal prompting with Vertex AI notebook](./gemini-multimodal.ipynb) and example experiment file in [data/input/gemini-multimodal-example.jsonl](https://github.com/alan-turing-institute/prompto/blob/main/examples/gemini/data/input/gemini-multimodal-example.jsonl). You can run it with the following command:
```bash
prompto_run_experiment --file data/input/gemini-multimodal-example.jsonl --max-queries 30
```

## Environment variables

To run the experiment, you will need to set the following environment variables first:
```bash
export GEMINI_API_KEY=<YOUR-GEMINI-KEY>
```

You can also use an `.env` file to save these environment variables without needing to export them globally in the terminal:
```
GEMINI_API_KEY=<YOUR-GEMINI-KEY>
```

By default, the `prompto_run_experiment` command will look for an `.env` file in the current directory. If you want to use a different `.env` file, you can specify it with the `--env` flag.

Also see the [gemini.ipynb](./gemini.ipynb) notebook for a more detailed walkthrough on the how to set the environment variables and run the experiment and the different types of prompts you can run.

Do note that when you run the experiment, the input file ([data/input/gemini-example.jsonl](https://github.com/alan-turing-institute/prompto/blob/main/examples/gemini/data/input/gemini-example.jsonl)) will be moved to the output directory (timestamped for when you run the experiment).
