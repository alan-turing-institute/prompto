# Contribute to `prompto`

Everyone is welcome to contribute to `prompto` and we value a wide range of contributions from code contributions or bug reports to documentation improvements. In particular, while `prompto` is a tool to support querying API endpoints asynchronously, there are several APIs that we have not implemented. We don't have access to every APIs and so we need your help to implement them! We are also open to new ideas and suggestions for the library.

This note aims to capture some of the practices adopted during the development of `prompto` with a view of making development easier and process of contributing to the library as smooth as possible.

It is not intended as a set of hard-and-fast rules * there will always be exceptions, and we definitely don't want to deter anyone from contributing, rather we hope that this will develop into a set of helpful guidelines, and additions/corrections to this document are always welcome!

Note that this guide was inspired by the [transformers guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).

### Sections

* [Ways to contribute](#ways-to-contribute)
* [Branching and making a pull request](#branching-and-making-a-pull-request)
* [Formatting and `pre-commit` hooks](#formatting-and-pre-commit-hooks)
* [Setting up a development environment](#setting-up-a-development-environment)

## Ways to contribute

There are many ways to contribute to `prompto`:

* Adding new APIs and models * see the [guide on how to add new APIs and models](add_new_api.md)
* Improving the documentation
* Contribute to the examples and creating tutorials
* Submitting bug reports or feature requests
* Fixing bugs or implementing new features

To see any open issues or to submit a new issue, you can visit the [issues page](https://github.com/alan-turing-institute/prompto/issues).

## Branching and making a pull request

Development should mostly take place on individual branches that are branched off `main`. When you are ready to merge your changes, you can create a pull request to merge your branch into `main`.

To start contributing, you can follow these steps:

1. **Fork the repository**
    You can fork the repository by clicking on the [**Fork**](https://github.com/alan-turing-institute/prompto/fork) button on the top right of the [repository page](https://github.com/alan-turing-institute/prompto).

2. **Clone your fork of repository**
    ```bash
    git clone git@github.com:<your Github handle>/prompto.git
    cd prompto
    git remote add upstream https://github.com/alan-turing-institute/prompto.git
    ```

3. **Create a development environment in a new virtual environment** * see [setting up a development environment](#setting-up-a-development-environment) section for more details.

3. **Create a new branch for your changes**
    ```bash
    git checkout -b my-new-feature
    ```

4. **Make your changes**
    You may also want to make sure your code is up-to-date with the original repository by rebasing your branch before making a pull request:
    ```bash
    git fetch upstream
    git rebase upstream/main
    ```

5. **Ensure that your changes are tested**
    You can run the tests using:
    ```bash
    python -m pytest
    ```

6. **Install pre-commit hooks** * see [formatting and `pre-commit` hooks](#formatting-and-pre-commit-hooks) section for more details.
    You can install the pre-commit hooks by running:
    ```bash
    pre-commit install
    ```

7. **Commit your changes and open a _Pull Request_**
    ```bash
    git add <files-with-your-changes>
    git commit -m "Add new feature"
    ```

    Do remember to write good commit messages to communicate your changes.

8. **Push your changes to your fork and make a pull request**
    Note may also want to make sure your code is up-to-date with the original repository by rebasing your branch before making a pull request:
    ```bash
    git fetch upstream
    git rebase upstream/main
    ```

    Then push your changes to your fork:
    ```bash
    git push -u origin my-new-feature
    ```

    You can now go to your fork of the repository on Github and create a pull request from your branch to the `main` branch of the original repository.

## Formatting and `pre-commit` hooks

We use `pre-commit` which will automatically format your code and run some basic checks before you commit to ensure that the code is formatted correctly. You can install the pre-commit hooks by running:
```bash
pip install pre-commit  # or brew install pre-commit on macOS
pre-commit install  # will install a pre-commit hook into the git repo
```

After doing this, each time you commit, some linters will be applied to format the codebase. You can also/alternatively run `pre-commit run --all-files` to run the checks.

## Setting up a development environment

If you'd like to set up a development environment for `prompto`, you can follow the steps below:

1. **Clone the repository** (or a fork of the repository if you are planning to contribute to the library * see the [branching and making a pull request](#branching-and-making-a-pull-request) section for more details)
    ```bash
    git clone git@github.com:alan-turing-institute/prompto.git
    ```

2. **Navigate to Project Directory**
    ```bash
    cd prompto
    ```

3. **Set up a development environment** (in a virtual environment)
    ```bash
    pip install -e ".[dev]"
    ```

    This will install the dependencies required for development. Note that there are groups for different models that you can install as well. For example, if you want to install the dependencies for the OpenAI and Gemini models, you can run:
    ```bash
    pip install -e ".[dev,openai,gemini]"
    ```

### Using poetry

`prompto` uses [Poetry](https://python-poetry.org/) for dependency management. If you prefer to use Poetry for dependency management, you can instead run the following to set up the development environment:

1. **Install Poetry**
    If you haven't installed Poetry yet, you can install it by following the instructions [here](https://python-poetry.org/docs/#installation).

2. **Create and activate a Poetry Environment**
    ```bash
    poetry shell
    ```

    This will create a virtual environment and activate it. You can also use another virtual environment manager, such as `venv` or `conda` for this step if you prefer.

3. **Install dependencies**
    ```bash
    poetry install --extras dev
    ```

    To install further groups, you can use multiple `--extras` or `-E` flags, or you can specify them in quotes:
    ```bash
    poetry install --extras "dev openai gemini"
    ```
    or
    ```bash
    poetry install -E dev -E openai -E gemini
    ```

    Refer to the [Poetry documentation](https://python-poetry.org/docs/cli/#install) for more information on installing dependencies.
