# Examples

There are several examples to guide you through the usage of the library:

* [Running experiments](./notebooks/running_experiments.ipynb) provides a walkthrough of key concepts and the main classes in the `prompto` library as well as a simple example of running an experiment using the `prompto` commands.
* [Grouping prompts and specifying rate limits](./notebooks/grouping_prompts_and_specifying_rate_limits.ipynb) provides a walkthrough of how parallel processing can be used to further speed up the querying of LLM endpoints. It shows how we to specify rate limits for different APIs or custom grouping of prompts.
* [System Demonstration](./system-demo/README.md) provides a series of small examples to compare `prompto` versus using traditional, synchronous API calls.

There are also specific examples for different APIs:

* [Azure OpenAI](./azure-openai/README.md)
* [OpenAI](./openai/README.md)
* [Anthropic](./anthropic/README.md)
* [Gemini](./gemini/README.md)
* [Vertex AI](./vertexai/README.md)
* [Ollama](./ollama/README.md)

Note that if you're viewing this page on the documentation pages, you might want to visit the [GitHub repository](https://github.com/alan-turing-institute/prompto/tree/main/examples) to access the examples and notebooks directly where you can see the folder structure of the [pipeline data folders](./../docs/pipeline.md) for each example.
