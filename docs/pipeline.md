# prompto Pipeline

The library has functionality to process experiments and to run a pipeline which continually looks for new experiment jsonl files in the input folder. Everything starts with defining a **pipeline data folder**, e.g. "data" which contains:
```
└── data
    └── input: contains the jsonl files with the experiments
    └── output: where the results of the experiments will be stored. When an experiment is ran, a folder is created within the output folder of the experiment  name (as defined in the jsonl file but removing the `.jsonl` extension) and the results and logs for the experiment are stored there
    └── media: which contains the media files for the experiments. These files must be within folders of the same experiment name (as defined in the jsonl file but removing the `.jsonl` extension)
```

## Running the pipeline

Once you have added the jsonl files to the `input/` folder of the data folder, you can run the pipeline process using the `prompto_run_pipeline` command in the terminal as follows:
```bash
prompto_run_pipeline --data-folder data
```

This initialises the process of continually checking the input folder for new experiments to process. If an experiment is found, it is processed and the results are stored in the output folder. The pipeline will continue to check for new experiments until the process is stopped.

If there are several experiments in the input folder, the pipeline will process the experiments in the order that the files were created/modified in the input folder (i.e. the oldest file will be processed first). This ordering is computed by using `os.path.getctime` which on some systems (e.g. Unix) is the time of the last metadata change and for tohers (e.g. Windows) is the creation time of the path.

## Run a single experiment

If you want to run a single experiment, you can use the `prompto_run_experiment` command in the terminal as follows:
```bash
prompto_run_experiment --file path/to/experiment.jsonl --data-folder data
```

This will process the experiment defined in the jsonl file and store the results in the output folder. Note that the path to the file doesn't neceesarily have to be within the `input/` folder of the data folder. In the case where it is not already in the `input/` folder, it will get moved there before processing and the output will be saved in the output folder as usual.

## Pipeline settings

When running the pipeline or an experiment, there are certain settings to define how to run the experiments. These can be set using the above command line interfaces via the following argument flags:
- `--data-folder` or `-d`: the path to the data folder which contains the input, output and media folders for the experiments (by default, `./data`)
- `--max-queries` or `-m`: the maximum number of queries to send within a minute (i.e. the query rate limit) (by default, `10`)
- `--max-attempts` or `-a`: the maximum number of attempts to try querying the model before giving up (by default, `5`)
- `--parallel` or `-p`: when the experiment files has different APIs to query, this flag allows the pipeline to send the queries to the different APIs in parallel (by default, `False`)

For example, to run the pipeline in `pipline-data/`, with a maximum of 5 queries per minute, have a maximum of 3 attempts for each query, and to send calls to separate API endpoints in parallel, you can run:
```bash
prompto_run_pipeline --data-folder pipeline-data --max-queries 5 --max-attempts 3 --parallel
```
and similarly for running a single experiment:
```bash
prompto_run_experiment --file path/to/experiment.jsonl --data-folder pipeline-data --max-queries 5 --max-attempts 3 --parallel
```
