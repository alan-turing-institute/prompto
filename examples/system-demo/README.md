# System Demonstration examples

We provide some illustrative examples of how to use `prompto` and compare it against traditional a synchronous approach to querying LLM endpoints.

We sample prompts from the instruction-following data following the Self-Instruct approach of [1] and [2]. We take a sample of 100 prompts from the [instruction-following data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) from [2] and apply the same prompt template. We then use these as prompt inputs to different models using `prompto`. See the [Generating the prompts for experiments](./alpaca_sample_generation.ipynb) notebook for more details.

We then have a series of different settings to illustrate the performance of `prompto` compared to a synchronous Python for loop:

- [Querying different LLM endpoints: `prompto` vs. synchronous Python for loop](./experiment_1.ipynb)
- [Querying different LLM endpoints: `prompto` _with parallel processing_ vs. synchronous Python for loop](./experiment_2.ipynb)
- [Querying different models from the same endpoint: `prompto` vs. synchronous Python for loop](./experiment_3.ipynb)

### References

[1]: Self-Instruct: Aligning Language Model with Self Generated Instructions. Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. 2022. https://arxiv.org/abs/2212.10560

[2]: Stanford Alpaca: An Instruction-following LLaMA model. Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto. 2023. https://github.com/tatsu-lab/stanford_alpaca.
