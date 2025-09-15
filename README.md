# [Open WebUI](https://docs.openwebui.com/getting-started/) pipelines

## Description

[`Pipelines`](https://docs.openwebui.com/pipelines/) functionality implementation.

## Project structure

```bash
├── Dockerfile                                  
├── pipelines                                   # here all pipelines should be defined
│   ├── example.env
│   ├── magsynth_pipeline                       # automatically created by OpenWebUI
│   │   └── valves.json
│   ├── magsynth_pipeline.py                    # working example of a Pipeline
│   ├── magsynth-dev.env                        # working example of an `.env` file for a Pipeline
│   └── utils
│       ├── core
│       │   ├── __init__.py
│       │   ├── abstract_pipeline.py            # here the Pipeline abstract class is defined
│       │   └── schemas.py
│       └── magsynth                            # <pipeline_name> pattern directory for the Pipeline utility files
│           ├── prompts
│           │   ├── __init__.py
│           │   ├── general_query_answer.py
│           │   └── query_categorization.py
│           └── schemas.py
├── pyproject.toml
├── README.md
└── uv.lock
```

Pipelines are an abstraction that allows to implement your own message processing steps (e.g. building RAG system)
and connect it seamlessly to OpenWebUI instance in the form of basic LLM chat.

## Usage

All dependencies are managed with the [uv](https://docs.astral.sh/uv/) dependency manager.

1. Create `<pipeline_name>_pipeline.py` file inside the `pipelines` directory.
2. Inside the file define your pipeline class and inherit it from the `AbstractPipeline` class.

    The `on_startup` method is here for loading all resources (such as HF models) and establishing connections to external 
    services.

    `on_shutdown` should be used for gently closing all connections and freeing resources if needed.

    The `pipe` method is the main messages entry point where all the processing logic must be implemented.

    You may define any number of methods you want.

3. Build the project with Docker.
