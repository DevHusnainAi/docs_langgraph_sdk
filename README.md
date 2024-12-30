
<a name="readmemd"></a>

## Setup

To setup requirements for building docs you can run:

```bash
poetry install --with test
```

### Serving documentation locally

To run the documentation server locally you can run:

```bash
make serve-docs
```

### Execute notebooks

If you would like to automatically execute all of the notebooks, to mimic the "Run notebooks" GHA, you can run:

```bash
python docs/_scripts/prepare_notebooks_for_ci.py
./docs/_scripts/execute_notebooks.sh
```

**Note**: if you want to run the notebooks without `%pip install` cells, you can run:

```bash
python docs/_scripts/prepare_notebooks_for_ci.py --comment-install-cells
./docs/_scripts/execute_notebooks.sh
```

`prepare_notebooks_for_ci.py` script will add VCR cassette context manager for each cell in the notebook, so that:
* when the notebook is run for the first time, cells with network requests will be recorded to a VCR cassette file
* when the notebook is run subsequently, the cells with network requests will be replayed from the cassettes

**Note**: this is currently limited only to the notebooks in `docs/docs/how-tos`

### Adding new notebooks

If you are adding a notebook with API requests, it's **recommended** to record network requests so that they can be subsequently replayed. If this is not done, the notebook runner will make API requests every time the notebook is run, which can be costly and slow.

To record network requests, please make sure to first run `prepare_notebooks_for_ci.py` script.

Then, run

```bash
jupyter execute <path_to_notebook>
```

Once the notebook is executed, you should see the new VCR cassettes recorded in `docs/cassettes` directory and discard the updated notebook.

### Updating existing notebooks

If you are updating an existing notebook, please make sure to remove any existing cassettes for the notebook in `docs/cassettes` directory (each cassette is prefixed with the notebook name), and then run the steps from the "Adding new notebooks" section above.

To delete cassettes for a notebook, you can run:

```bash
rm docs/cassettes/<notebook_name>*
```

<a name="docsclouddeploymentcloudmd"></a>

## How to Deploy to LangGraph Cloud

LangGraph Cloud is available within <a href="https://www.langchain.com/langsmith" target="_blank">LangSmith</a>. To deploy a LangGraph Cloud API, navigate to the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>.

### Prerequisites

1. LangGraph Cloud applications are deployed from GitHub repositories. Configure and upload a LangGraph Cloud application to a GitHub repository in order to deploy it to LangGraph Cloud.
1. [Verify that the LangGraph API runs locally](#docsclouddeploymenttest_locallymd). If the API does not run successfully (i.e. `langgraph dev`), deploying to LangGraph Cloud will fail as well.

### Create New Deployment

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `LangGraph Cloud`. The `LangGraph Cloud` view contains a list of existing LangGraph Cloud deployments.
1. In the top-right corner, select `+ New Deployment` to create a new deployment.
1. In the `Create New Deployment` panel, fill out the required fields.
    1. `Deployment details`
        1. Select `Import from GitHub` and follow the GitHub OAuth workflow to install and authorize LangChain's `hosted-langserve` GitHub app to access the selected repositories. After installation is complete, return to the `Create New Deployment` panel and select the GitHub repository to deploy from the dropdown menu. **Note**: The GitHub user installing LangChain's `hosted-langserve` GitHub app must be an [owner](https://docs.github.com/en/organizations/managing-peoples-access-to-your-organization-with-roles/roles-in-an-organization#organization-owners) of the organization or account.
        1. Specify a name for the deployment.
        1. Specify the desired `Git Branch`. A deployment is linked to a branch. When a new revision is created, code for the linked branch will be deployed. The branch can be updated later in the [Deployment Settings](#deployment-settings).
        1. Specify the full path to the [LangGraph API config file](#configuration-file) including the file name. For example, if the file `langgraph.json` is in the root of the repository, simply specify `langgraph.json`.
        1. Check/uncheck checkbox to `Automatically update deployment on push to branch`. If checked, the deployment will automatically be updated when changes are pushed to the specified `Git Branch`. This setting can be enabled/disabled later in the [Deployment Settings](#deployment-settings).
    1. Select the desired `Deployment Type`.
        1. `Development` deployments are meant for non-production use cases and are provisioned with minimal resources.
        1. `Production` deployments can serve up to 500 requests/second and are provisioned with highly available storage with automatic backups.
    1. Determine if the deployment should be `Shareable through LangGraph Studio`.
        1. If unchecked, the deployment will only be accessible with a valid LangSmith API key for the workspace.
        1. If checked, the deployment will be accessible through LangGraph Studio to any LangSmith user. A direct URL to LangGraph Studio for the deployment will be provided to share with other LangSmith users.
    1. Specify `Environment Variables` and secrets. See the [Environment Variables reference](#docscloudreferenceenv_varmd) to configure additional variables for the deployment.
        1. Sensitive values such as API keys (e.g. `OPENAI_API_KEY`) should be specified as secrets.
        1. Additional non-secret environment variables can be specified as well.
    1. A new LangSmith `Tracing Project` is automatically created with the same name as the deployment.
1. In the top-right corner, select `Submit`. After a few seconds, the `Deployment` view appears and the new deployment will be queued for provisioning.

### Create New Revision

When [creating a new deployment](#create-new-deployment), a new revision is created by default. Subsequent revisions can be created to deploy new code changes.

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `LangGraph Cloud`. The `LangGraph Cloud` view contains a list of existing LangGraph Cloud deployments.
1. Select an existing deployment to create a new revision for.
1. In the `Deployment` view, in the top-right corner, select `+ New Revision`.
1. In the `New Revision` modal, fill out the required fields.
    1. Specify the full path to the [LangGraph API config file](#configuration-file) including the file name. For example, if the file `langgraph.json` is in the root of the repository, simply specify `langgraph.json`.
    1. Determine if the deployment should be `Shareable through LangGraph Studio`.
        1. If unchecked, the deployment will only be accessible with a valid LangSmith API key for the workspace.
        1. If checked, the deployment will be accessible through LangGraph Studio to any LangSmith user. A direct URL to LangGraph Studio for the deployment will be provided to share with other LangSmith users.
    1. Specify `Environment Variables` and secrets. Existing secrets and environment variables are prepopulated. See the [Environment Variables reference](#docscloudreferenceenv_varmd) to configure additional variables for the revision.
        1. Add new secrets or environment variables.
        1. Remove existing secrets or environment variables.
        1. Update the value of existing secrets or environment variables.
1. Select `Submit`. After a few seconds, the `New Revision` modal will close and the new revision will be queued for deployment.

### View Build and Deployment Logs

Build and deployment logs are available for each revision.

Starting from the `LangGraph Cloud` view...

1. Select the desired revision from the `Revisions` table. A panel slides open from the right-hand side and the `Build` tab is selected by default, which displays build logs for the revision.
1. In the panel, select the `Deploy` tab to view deployment logs for the revision.
1. Within the `Deploy` tab, adjust the date/time range picker as needed. By default, the date/time range picker is set to the `Last 15 minutes`.

### Interrupt Revision

Interrupting a revision will stop deployment of the revision.

!!! warning "Undefined Behavior"
    Interrupted revisions have undefined behavior. This is only useful if you need to deploy a new revision and you already have a revision "stuck" in progress. In the future, this feature may be removed.

Starting from the `LangGraph Cloud` view...

1. Select the menu icon (three dots) on the right-hand side of the row for the desired revision from the `Revisions` table.
1. Select `Interrupt` from the menu.
1. A modal will appear. Review the confirmation message. Select `Interrupt revision`.

### Delete Deployment

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `LangGraph Cloud`. The `LangGraph Cloud` view contains a list of existing LangGraph Cloud deployments.
1. Select the menu icon (three dots) on the right-hand side of the row for the desired deployment and select `Delete`.
1. A `Confirmation` modal will appear. Select `Delete`.

### Deployment Settings

Starting from the `LangGraph Cloud` view...

1. In the top-right corner, select the gear icon (`Deployment Settings`).
1. Update the `Git Branch` to the desired branch.
1. Check/uncheck checkbox to `Automatically update deployment on push to branch`.
    1. Branch creation/deletion and tag creation/deletion events will not trigger an update. Only pushes to an existing branch will trigger an update.
    1. Pushes in quick succession to a branch will not trigger subsequent updates. In the future, this functionality may be changed/improved.


<a name="docsclouddeploymentcustom_dockermd"></a>

## How to customize Dockerfile

Users can add an array of additional lines to add to the Dockerfile following the import from the parent LangGraph image. In order to do this, you simply need to modify your `langgraph.json` file by passing in the commands you want run to the `dockerfile_lines` key. For example, if we wanted to use `Pillow` in our graph you would need to add the following dependencies:

```
{
    "dependencies": ["."],
    "graphs": {
        "openai_agent": "./openai_agent.py:agent",
    },
    "env": "./.env",
    "dockerfile_lines": [
        "RUN apt-get update && apt-get install -y libjpeg-dev zlib1g-dev libpng-dev",
        "RUN pip install Pillow"
    ]
}
```

This would install the system packages required to use Pillow if we were working with `jpeq` or `png` image formats. 

<a name="docsclouddeploymentgraph_rebuildmd"></a>

## Rebuild Graph at Runtime

You might need to rebuild your graph with a different configuration for a new run. For example, you might need to use a different graph state or graph structure depending on the config. This guide shows how you can do this.

!!! note "Note"
    In most cases, customizing behavior based on the config should be handled by a single graph where each node can read a config and change its behavior based on it

### Prerequisites

Make sure to check out [this how-to guide](#docsclouddeploymentsetupmd) on setting up your app for deployment first.

### Define graphs

Let's say you have an app with a simple graph that calls an LLM and returns the response to the user. The app file directory looks like the following:

```
my-app/
|-- requirements.txt
|-- .env
|-- openai_agent.py     # code for your graph
```

where the graph is defined in `openai_agent.py`. 

#### No rebuild

In the standard LangGraph API configuration, the server uses the compiled graph instance that's defined at the top level of `openai_agent.py`, which looks like the following:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph

model = ChatOpenAI(temperature=0)

graph_workflow = MessageGraph()

graph_workflow.add_node("agent", model)
graph_workflow.add_edge("agent", END)
graph_workflow.add_edge(START, "agent")

agent = graph_workflow.compile()
```

To make the server aware of your graph, you need to specify a path to the variable that contains the `CompiledStateGraph` instance in your LangGraph API configuration (`langgraph.json`), e.g.:

```
{
    "dependencies": ["."],
    "graphs": {
        "openai_agent": "./openai_agent.py:agent",
    },
    "env": "./.env"
}
```

#### Rebuild

To make your graph rebuild on each new run with custom configuration, you need to rewrite `openai_agent.py` to instead provide a _function_ that takes a config and returns a graph (or compiled graph) instance. Let's say we want to return our existing graph for user ID '1', and a tool-calling agent for other users. We can modify `openai_agent.py` as follows:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


model = ChatOpenAI(temperature=0)

def make_default_graph():
    """Make a simple LLM agent"""
    graph_workflow = StateGraph(State)
    def call_model(state):
        return {"messages": [model.invoke(state["messages"])]}

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()
    return agent


def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent


## this is the graph making function that will decide which graph to
## build based on the provided config
def make_graph(config: RunnableConfig):
    user_id = config.get("configurable", {}).get("user_id")
    # route to different graph state / structure based on the user ID
    if user_id == "1":
        return make_default_graph()
    else:
        return make_alternative_graph()
```

Finally, you need to specify the path to your graph-making function (`make_graph`) in `langgraph.json`:

```
{
    "dependencies": ["."],
    "graphs": {
        "openai_agent": "./openai_agent.py:make_graph",
    },
    "env": "./.env"
}
```

See more info on LangGraph API configuration file [here](#configuration-file)

<a name="docsclouddeploymentsemantic_searchmd"></a>

## How to add semantic search to your LangGraph deployment

This guide explains how to add semantic search to your LangGraph deployment's cross-thread [store](#memory-store), so that your agent can search for memories and other documents by semantic similarity.

### Prerequisites

- A LangGraph deployment (see [how to deploy](#docsclouddeploymentsetup_pyprojectmd))
- API keys for your embedding provider (in this case, OpenAI)
- `langchain >= 0.3.8` (if you specify using the string format below)

### Steps

1. Update your `langgraph.json` configuration file to include the store configuration:

```json
{
    ...
    "store": {
        "index": {
            "embed": "openai:text-embeddings-3-small",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

This configuration:

- Uses OpenAI's text-embeddings-3-small model for generating embeddings
- Sets the embedding dimension to 1536 (matching the model's output)
- Indexes all fields in your stored data (`["$"]` means index everything, or specify specific fields like `["text", "metadata.title"]`)

2. To use the string embedding format above, make sure your dependencies include `langchain >= 0.3.8`:

```toml
## In pyproject.toml
[project]
dependencies = [
    "langchain>=0.3.8"
]
```

Or if using requirements.txt:

```
langchain>=0.3.8
```

### Usage

Once configured, you can use semantic search in your LangGraph nodes. The store requires a namespace tuple to organize memories:

```python
def search_memory(state: State, *, store: BaseStore):
    # Search the store using semantic similarity
    # The namespace tuple helps organize different types of memories
    # e.g., ("user_facts", "preferences") or ("conversation", "summaries")
    results = store.search(
        namespace=("memory", "facts"),  # Organize memories by type
        query="your search query",
        limit=3  # number of results to return
    )
    return results
```

### Custom Embeddings

If you want to use custom embeddings, you can pass a path to a custom embedding function:

```json
{
    ...
    "store": {
        "index": {
            "embed": "path/to/embedding_function.py:embed",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

The deployment will look for the function in the specified path. The function must be async and accept a list of strings:

```python
## path/to/embedding_function.py
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def aembed_texts(texts: list[str]) -> list[list[float]]:
    """Custom embedding function that must:
    1. Be async
    2. Accept a list of strings
    3. Return a list of float arrays (embeddings)
    """
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]
```

### Querying via the API

You can also query the store using the LangGraph SDK. Since the SDK uses async operations:

```python
from langgraph_sdk import get_client

async def search_store():
    client = get_client()
    results = await client.store.search_items(
        ("memory", "facts"),
        query="your search query",
        limit=3  # number of results to return
    )
    return results

## Use in an async context
results = await search_store()
```


<a name="docsclouddeploymentsetupmd"></a>

## How to Set Up a LangGraph Application for Deployment

A LangGraph application must be configured with a [LangGraph API configuration file](#configuration-file) in order to be deployed to LangGraph Cloud (or to be self-hosted). This how-to guide discusses the basic steps to setup a LangGraph application for deployment using `requirements.txt` to specify project dependencies.

This walkthrough is based on [this repository](https://github.com/langchain-ai/langgraph-example), which you can play around with to learn more about how to setup your LangGraph application for deployment.

!!! tip "Setup with pyproject.toml"
    If you prefer using poetry for dependency management, check out [this how-to guide](#docsclouddeploymentsetup_pyprojectmd) on using `pyproject.toml` for LangGraph Cloud.

!!! tip "Setup with a Monorepo"
    If you are interested in deploying a graph located inside a monorepo, take a look at [this](https://github.com/langchain-ai/langgraph-example-monorepo) repository for an example of how to do so.

The final repo structure will look something like this:

```bash
my-app/
Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
Ôöé   Ôö£ÔöÇÔöÇ utils # utilities for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ __init__.py
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.py # tools for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.py # node functions for you graph
Ôöé   Ôöé   ÔööÔöÇÔöÇ state.py # state definition of your graph
Ôöé┬á┬á Ôö£ÔöÇÔöÇ requirements.txt # package dependencies
Ôöé┬á┬á Ôö£ÔöÇÔöÇ __init__.py
Ôöé┬á┬á ÔööÔöÇÔöÇ agent.py # code for constructing your graph
Ôö£ÔöÇÔöÇ .env # environment variables
ÔööÔöÇÔöÇ langgraph.json # configuration file for LangGraph
```

After each step, an example file directory is provided to demonstrate how code can be organized.

### Specify Dependencies

Dependencies can optionally be specified in one of the following files: `pyproject.toml`, `setup.py`, or `requirements.txt`. If none of these files is created, then dependencies can be specified later in the [LangGraph API configuration file](#create-langgraph-api-config).

The dependencies below will be included in the image, you can also use them in your code, as long as with a compatible version range:

```
langgraph>=0.2.56,<0.3.0
langgraph-checkpoint>=2.0.5,<3.0
langchain-core>=0.2.38,<0.4.0
langsmith>=0.1.63
orjson>=3.9.7
httpx>=0.25.0
tenacity>=8.0.0
uvicorn>=0.26.0
sse-starlette>=2.1.0
uvloop>=0.18.0
httptools>=0.5.0
jsonschema-rs>=0.16.3
croniter>=1.0.1
structlog>=23.1.0
redis>=5.0.0,<6.0.0
```

Example `requirements.txt` file:

```
langgraph
langchain_anthropic
tavily-python
langchain_community
langchain_openai

```

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
Ôöé┬á┬á ÔööÔöÇÔöÇ requirements.txt # package dependencies
```

### Specify Environment Variables

Environment variables can optionally be specified in a file (e.g. `.env`). See the [Environment Variables reference](#docscloudreferenceenv_varmd) to configure additional variables for a deployment.

Example `.env` file:

```
MY_ENV_VAR_1=foo
MY_ENV_VAR_2=bar
OPENAI_API_KEY=key
```

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
Ôöé┬á┬á ÔööÔöÇÔöÇ requirements.txt # package dependencies
ÔööÔöÇÔöÇ .env # environment variables
```

### Define Graphs

Implement your graphs! Graphs can be defined in a single file or multiple files. Make note of the variable names of each [CompiledGraph][langgraph.graph.graph.CompiledGraph] to be included in the LangGraph application. The variable names will be used later when creating the [LangGraph API configuration file](#configuration-file).

Example `agent.py` file, which shows how to import from other modules you define (code for the modules is not shown here, please see [this repo](https://github.com/langchain-ai/langgraph-example) to see their implementation):

```python
## my_agent/agent.py
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import call_model, should_continue, tool_node # import nodes
from my_agent.utils.state import AgentState # import state

## Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

graph = workflow.compile()
```

!!! warning "Assign `CompiledGraph` to Variable"
    The build process for LangGraph Cloud requires that the `CompiledGraph` object be assigned to a variable at the top-level of a Python module (alternatively, you can provide [a function that creates a graph](#docsclouddeploymentgraph_rebuildmd)).

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
Ôöé   Ôö£ÔöÇÔöÇ utils # utilities for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ __init__.py
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.py # tools for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.py # node functions for you graph
Ôöé   Ôöé   ÔööÔöÇÔöÇ state.py # state definition of your graph
Ôöé┬á┬á Ôö£ÔöÇÔöÇ requirements.txt # package dependencies
Ôöé┬á┬á Ôö£ÔöÇÔöÇ __init__.py
Ôöé┬á┬á ÔööÔöÇÔöÇ agent.py # code for constructing your graph
ÔööÔöÇÔöÇ .env # environment variables
```

### Create LangGraph API Config

Create a [LangGraph API configuration file](#configuration-file) called `langgraph.json`. See the [LangGraph CLI reference](#configuration-file) for detailed explanations of each key in the JSON object of the configuration file.

Example `langgraph.json` file:

```json
{
  "dependencies": ["./my_agent"],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": ".env"
}
```

Note that the variable name of the `CompiledGraph` appears at the end of the value of each subkey in the top-level `graphs` key (i.e. `:<variable_name>`).

!!! warning "Configuration Location"
    The LangGraph API configuration file must be placed in a directory that is at the same level or higher than the Python files that contain compiled graphs and associated dependencies.

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
Ôöé   Ôö£ÔöÇÔöÇ utils # utilities for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ __init__.py
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.py # tools for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.py # node functions for you graph
Ôöé   Ôöé   ÔööÔöÇÔöÇ state.py # state definition of your graph
Ôöé┬á┬á Ôö£ÔöÇÔöÇ requirements.txt # package dependencies
Ôöé┬á┬á Ôö£ÔöÇÔöÇ __init__.py
Ôöé┬á┬á ÔööÔöÇÔöÇ agent.py # code for constructing your graph
Ôö£ÔöÇÔöÇ .env # environment variables
ÔööÔöÇÔöÇ langgraph.json # configuration file for LangGraph
```

### Next

After you setup your project and place it in a github repo, it's time to [deploy your app](#docsclouddeploymentcloudmd).


<a name="docsclouddeploymentsetup_javascriptmd"></a>

## How to Set Up a LangGraph.js Application for Deployment

A [LangGraph.js](https://langchain-ai.github.io/langgraphjs/) application must be configured with a [LangGraph API configuration file](#configuration-file) in order to be deployed to LangGraph Cloud (or to be self-hosted). This how-to guide discusses the basic steps to setup a LangGraph.js application for deployment using `package.json` to specify project dependencies.

This walkthrough is based on [this repository](https://github.com/langchain-ai/langgraphjs-studio-starter), which you can play around with to learn more about how to setup your LangGraph application for deployment.

The final repo structure will look something like this:

```bash
my-app/
Ôö£ÔöÇÔöÇ src # all project code lies within here
Ôöé   Ôö£ÔöÇÔöÇ utils # optional utilities for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.ts # tools for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.ts # node functions for you graph
Ôöé   Ôöé   ÔööÔöÇÔöÇ state.ts # state definition of your graph
Ôöé┬á┬á ÔööÔöÇÔöÇ agent.ts # code for constructing your graph
Ôö£ÔöÇÔöÇ package.json # package dependencies
Ôö£ÔöÇÔöÇ .env # environment variables
ÔööÔöÇÔöÇ langgraph.json # configuration file for LangGraph
```

After each step, an example file directory is provided to demonstrate how code can be organized.

### Specify Dependencies

Dependencies can be specified in a `package.json`. If none of these files is created, then dependencies can be specified later in the [LangGraph API configuration file](#create-langgraph-api-config).

Example `package.json` file:

```json
{
  "name": "langgraphjs-studio-starter",
  "packageManager": "yarn@1.22.22",
  "dependencies": {
    "@langchain/community": "^0.2.31",
    "@langchain/core": "^0.2.31",
    "@langchain/langgraph": "^0.2.0",
    "@langchain/openai": "^0.2.8"
  }
}
```

Example file directory:

```bash
my-app/
ÔööÔöÇÔöÇ package.json # package dependencies
```

### Specify Environment Variables

Environment variables can optionally be specified in a file (e.g. `.env`). See the [Environment Variables reference](#docscloudreferenceenv_varmd) to configure additional variables for a deployment.

Example `.env` file:

```
MY_ENV_VAR_1=foo
MY_ENV_VAR_2=bar
OPENAI_API_KEY=key
TAVILY_API_KEY=key_2
```

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ package.json
ÔööÔöÇÔöÇ .env # environment variables
```

### Define Graphs

Implement your graphs! Graphs can be defined in a single file or multiple files. Make note of the variable names of each compiled graph to be included in the LangGraph application. The variable names will be used later when creating the [LangGraph API configuration file](#configuration-file).

Here is an example `agent.ts`:

```ts
import type { AIMessage } from "@langchain/core/messages";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatOpenAI } from "@langchain/openai";

import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

const tools = [
  new TavilySearchResults({ maxResults: 3, }),
];

// Define the function that calls the model
async function callModel(
  state: typeof MessagesAnnotation.State,
) {
  /**
   * Call the LLM powering our agent.
   * Feel free to customize the prompt, model, and other logic!
   */
  const model = new ChatOpenAI({
    model: "gpt-4o",
  }).bindTools(tools);

  const response = await model.invoke([
    {
      role: "system",
      content: `You are a helpful assistant. The current date is ${new Date().getTime()}.`
    },
    ...state.messages
  ]);

  // MessagesAnnotation supports returning a single message or array of messages
  return { messages: response };
}

// Define the function that determines whether to continue or not
function routeModelOutput(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const lastMessage: AIMessage = messages[messages.length - 1];
  // If the LLM is invoking tools, route there.
  if ((lastMessage?.tool_calls?.length ?? 0) > 0) {
    return "tools";
  }
  // Otherwise end the graph.
  return "__end__";
}

// Define a new graph.
// See https://langchain-ai.github.io/langgraphjs/how-tos/define-state/#getting-started for
// more on defining custom graph states.
const workflow = new StateGraph(MessagesAnnotation)
  // Define the two nodes we will cycle between
  .addNode("callModel", callModel)
  .addNode("tools", new ToolNode(tools))
  // Set the entrypoint as `callModel`
  // This means that this node is the first one called
  .addEdge("__start__", "callModel")
  .addConditionalEdges(
    // First, we define the edges' source node. We use `callModel`.
    // This means these are the edges taken after the `callModel` node is called.
    "callModel",
    // Next, we pass in the function that will determine the sink node(s), which
    // will be called after the source node is called.
    routeModelOutput,
    // List of the possible destinations the conditional edge can route to.
    // Required for conditional edges to properly render the graph in Studio
    [
      "tools",
      "__end__"
    ],
  )
  // This means that after `tools` is called, `callModel` node is called next.
  .addEdge("tools", "callModel");

// Finally, we compile it!
// This compiles it into a graph you can invoke and deploy.
export const graph = workflow.compile();
```

!!! info "Assign `CompiledGraph` to Variable"
    The build process for LangGraph Cloud requires that the `CompiledGraph` object be assigned to a variable at the top-level of a JavaScript module (alternatively, you can provide [a function that creates a graph](#docsclouddeploymentgraph_rebuildmd)).

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ src # all project code lies within here
Ôöé   Ôö£ÔöÇÔöÇ utils # optional utilities for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.ts # tools for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.ts # node functions for you graph
Ôöé   Ôöé   ÔööÔöÇÔöÇ state.ts # state definition of your graph
Ôöé┬á┬á ÔööÔöÇÔöÇ agent.ts # code for constructing your graph
Ôö£ÔöÇÔöÇ package.json # package dependencies
Ôö£ÔöÇÔöÇ .env # environment variables
ÔööÔöÇÔöÇ langgraph.json # configuration file for LangGraph
```

### Create LangGraph API Config

Create a [LangGraph API configuration file](#configuration-file) called `langgraph.json`. See the [LangGraph CLI reference](#configuration-file) for detailed explanations of each key in the JSON object of the configuration file.

Example `langgraph.json` file:

```json
{
  "node_version": "20",
  "dockerfile_lines": [],
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent.ts:graph"
  },
  "env": ".env"
}
```

Note that the variable name of the `CompiledGraph` appears at the end of the value of each subkey in the top-level `graphs` key (i.e. `:<variable_name>`).

!!! info "Configuration Location"
    The LangGraph API configuration file must be placed in a directory that is at the same level or higher than the TypeScript files that contain compiled graphs and associated dependencies.

### Next

After you setup your project and place it in a github repo, it's time to [deploy your app](#docsclouddeploymentcloudmd).


<a name="docsclouddeploymentsetup_pyprojectmd"></a>

## How to Set Up a LangGraph Application for Deployment

A LangGraph application must be configured with a [LangGraph API configuration file](#configuration-file) in order to be deployed to LangGraph Cloud (or to be self-hosted). This how-to guide discusses the basic steps to setup a LangGraph application for deployment using `pyproject.toml` to define your package's dependencies.

This walkthrough is based on [this repository](https://github.com/langchain-ai/langgraph-example-pyproject), which you can play around with to learn more about how to setup your LangGraph application for deployment.

!!! tip "Setup with requirements.txt"
    If you prefer using `requirements.txt` for dependency management, check out [this how-to guide](#docsclouddeploymentsetupmd).

!!! tip "Setup with a Monorepo"
    If you are interested in deploying a graph located inside a monorepo, take a look at [this](https://github.com/langchain-ai/langgraph-example-monorepo) repository for an example of how to do so.

The final repo structure will look something like this:

```bash
my-app/
Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
Ôöé   Ôö£ÔöÇÔöÇ utils # utilities for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ __init__.py
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.py # tools for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.py # node functions for you graph
Ôöé   Ôöé   ÔööÔöÇÔöÇ state.py # state definition of your graph
Ôöé┬á┬á Ôö£ÔöÇÔöÇ __init__.py
Ôöé┬á┬á ÔööÔöÇÔöÇ agent.py # code for constructing your graph
Ôö£ÔöÇÔöÇ .env # environment variables
Ôö£ÔöÇÔöÇ langgraph.json  # configuration file for LangGraph
ÔööÔöÇÔöÇ pyproject.toml # dependencies for your project
```

After each step, an example file directory is provided to demonstrate how code can be organized.

### Specify Dependencies

Dependencies can optionally be specified in one of the following files: `pyproject.toml`, `setup.py`, or `requirements.txt`. If none of these files is created, then dependencies can be specified later in the [LangGraph API configuration file](#create-langgraph-api-config).

The dependencies below will be included in the image, you can also use them in your code, as long as with a compatible version range:

```
langgraph>=0.2.56,<0.3.0
langgraph-checkpoint>=2.0.5,<3.0
langchain-core>=0.2.38,<0.4.0
langsmith>=0.1.63
orjson>=3.9.7
httpx>=0.25.0
tenacity>=8.0.0
uvicorn>=0.26.0
sse-starlette>=2.1.0
uvloop>=0.18.0
httptools>=0.5.0
jsonschema-rs>=0.16.3
croniter>=1.0.1
structlog>=23.1.0
redis>=5.0.0,<6.0.0
```

Example `pyproject.toml` file:

```toml
[tool.poetry]
name = "my-agent"
version = "0.0.1"
description = "An excellent agent build for LangGraph cloud."
authors = ["Polly the parrot <1223+polly@users.noreply.github.com>"]
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.9.0,<3.13"
langgraph = "^0.2.0"
langchain-fireworks = "^0.1.3"


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

Example file directory:

```bash
my-app/
ÔööÔöÇÔöÇ pyproject.toml   # Python packages required for your graph
```

### Specify Environment Variables

Environment variables can optionally be specified in a file (e.g. `.env`). See the [Environment Variables reference](#docscloudreferenceenv_varmd) to configure additional variables for a deployment.

Example `.env` file:

```
MY_ENV_VAR_1=foo
MY_ENV_VAR_2=bar
FIREWORKS_API_KEY=key
```

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ .env             # file with environment variables
ÔööÔöÇÔöÇ pyproject.toml
```

### Define Graphs

Implement your graphs! Graphs can be defined in a single file or multiple files. Make note of the variable names of each [CompiledGraph][langgraph.graph.graph.CompiledGraph] to be included in the LangGraph application. The variable names will be used later when creating the [LangGraph API configuration file](#configuration-file).

Example `agent.py` file, which shows how to import from other modules you define (code for the modules is not shown here, please see [this repo](https://github.com/langchain-ai/langgraph-example-pyproject) to see their implementation):

```python
## my_agent/agent.py
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from my_agent.utils.nodes import call_model, should_continue, tool_node # import nodes
from my_agent.utils.state import AgentState # import state

## Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

graph = workflow.compile()
```

!!! warning "Assign `CompiledGraph` to Variable"
    The build process for LangGraph Cloud requires that the `CompiledGraph` object be assigned to a variable at the top-level of a Python module.

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
Ôöé   Ôö£ÔöÇÔöÇ utils # utilities for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ __init__.py
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.py # tools for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.py # node functions for you graph
Ôöé   Ôöé   ÔööÔöÇÔöÇ state.py # state definition of your graph
Ôöé┬á┬á Ôö£ÔöÇÔöÇ __init__.py
Ôöé┬á┬á ÔööÔöÇÔöÇ agent.py # code for constructing your graph
Ôö£ÔöÇÔöÇ .env
ÔööÔöÇÔöÇ pyproject.toml
```

### Create LangGraph API Config

Create a [LangGraph API configuration file](#configuration-file) called `langgraph.json`. See the [LangGraph CLI reference](#configuration-file) for detailed explanations of each key in the JSON object of the configuration file.

Example `langgraph.json` file:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./my_agent/agent.py:graph"
  },
  "env": ".env"
}
```

Note that the variable name of the `CompiledGraph` appears at the end of the value of each subkey in the top-level `graphs` key (i.e. `:<variable_name>`).

!!! warning "Configuration Location"
    The LangGraph API configuration file must be placed in a directory that is at the same level or higher than the Python files that contain compiled graphs and associated dependencies.

Example file directory:

```bash
my-app/
Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
Ôöé   Ôö£ÔöÇÔöÇ utils # utilities for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ __init__.py
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.py # tools for your graph
Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.py # node functions for you graph
Ôöé   Ôöé   ÔööÔöÇÔöÇ state.py # state definition of your graph
Ôöé┬á┬á Ôö£ÔöÇÔöÇ __init__.py
Ôöé┬á┬á ÔööÔöÇÔöÇ agent.py # code for constructing your graph
Ôö£ÔöÇÔöÇ .env # environment variables
Ôö£ÔöÇÔöÇ langgraph.json  # configuration file for LangGraph
ÔööÔöÇÔöÇ pyproject.toml # dependencies for your project
```

### Next

After you setup your project and place it in a github repo, it's time to [deploy your app](#docsclouddeploymentcloudmd).


<a name="docsclouddeploymenttest_locallymd"></a>

## How to test a LangGraph app locally

This guide assumes you have a LangGraph app correctly set up with a proper configuration file and a corresponding compiled graph, and that you have a proper LangChain API key.

Testing locally ensures that there are no errors or conflicts with Python dependencies and confirms that the configuration file is specified correctly.

### Setup

Install the LangGraph CLI package:

```bash
pip install -U "langgraph-cli[inmem]"
```

Ensure you have an API key, which you can create from the [LangSmith UI](https://smith.langchain.com) (Settings > API Keys). This is required to authenticate that you have LangGraph Cloud access. After you have saved the key to a safe place, place the following line in your `.env` file:

```python
LANGSMITH_API_KEY = *********
```

### Start the API server

Once you have installed the CLI, you can run the following command to start the API server for local testing:

```shell
langgraph dev
```

This will start up the LangGraph API server locally. If this runs successfully, you should see something like:

>    Ready!
> 
>    - API: [http://localhost:2024](http://localhost:2024/)
>     
>    - Docs: http://localhost:2024/docs
>     
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

!!! note "In-Memory Mode"

    The `langgraph dev` command starts LangGraph Server in an in-memory mode. This mode is suitable for development and testing purposes. For production use, you should deploy LangGraph Server with access to a persistent storage backend.

    If you want to test your application with a persistent storage backend, you can use the `langgraph up` command instead of `langgraph dev`. You will
    need to have `docker` installed on your machine to use this command.


#### Interact with the server

We can now interact with the API server using the LangGraph SDK. First, we need to start our client, select our assistant (in this case a graph we called "agent", make sure to select the proper assistant you wish to test).

You can either initialize by passing authentication or by setting an environment variable.

##### Initialize with authentication

=== "Python"

    ```python
    from langgraph_sdk import get_client

    # only pass the url argument to get_client() if you changed the default port when calling langgraph dev
    client = get_client(url=<DEPLOYMENT_URL>,api_key=<LANGSMITH_API_KEY>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    // only set the apiUrl if you changed the default port when calling langgraph dev
    const client = new Client({ apiUrl: <DEPLOYMENT_URL>, apiKey: <LANGSMITH_API_KEY> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json'
      --header 'x-api-key: <LANGSMITH_API_KEY>'
    ```
  

##### Initialize with environment variables

If you have a `LANGSMITH_API_KEY` set in your environment, you do not need to explicitly pass authentication to the client

=== "Python"

    ```python
    from langgraph_sdk import get_client

    # only pass the url argument to get_client() if you changed the default port when calling langgraph dev
    client = get_client()
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    // only set the apiUrl if you changed the default port when calling langgraph dev
    const client = new Client();
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json'
    ```

Now we can invoke our graph to ensure it is working. Make sure to change the input to match the proper schema for your graph. 

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```
=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "what's the weather in sf"}] }

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf\"}]},
       \"stream_mode\": [
         \"events\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "") {
             print data_content "\n"
         }
         sub(/^event: /, "Receiving event of type: ", $0)
         printf "%s...\n", $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "") {
             print data_content "\n"
         }
     }
     ' 
    ```

If your graph works correctly, you should see your graph output displayed in the console. Of course, there are many more ways you might need to test your graph, for a full list of commands you can send with the SDK, see the [Python](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/) and [JS/TS](https://langchain-ai.github.io/langgraph/cloud/reference/sdk/js_ts_sdk_ref/) references.


<a name="docscloudhow-tosassistant_versioningmd"></a>

## How to version assistants

In this how-to guide we will walk through how you can create and manage different assistant versions. If you haven't already, you can read [this](#versioning-assistants) conceptual guide to gain a better understanding of what assistant versioning is. This how-to assumes you have a graph that is configurable, which means you have defined a config schema and passed it to your graph as follows:

=== "Python"

    ```python
    class Config(BaseModel):
        model_name: Literal["anthropic", "openai"] = "anthropic"
        system_prompt: str

    agent = StateGraph(State, config_schema=Config)
    ```

=== "Javascript"

    ```js
    const ConfigAnnotation = Annotation.Root({
        modelName: Annotation<z.enum(["openai", "anthropic"])>({
            default: () => "anthropic",
        }),
        systemPrompt: Annotation<String>
    });

    // the rest of your code

    const agent = new StateGraph(StateAnnotation, ConfigAnnotation);
    ```

### Setup

First let's set up our client and thread. If you are using the Studio, just open the application to the graph called "agent". If using cURL, you don't need to do anything except copy down your deployment URL and the name of the graph you want to use.

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    graph_name = "agent"
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const graphName = "agent";
    ```

### Create an assistant

For this example, we will create an assistant by modifying the model name that is used in our graph. We can create a new assistant called "openai_assistant" for this:

=== "Python"

    ```python
    openai_assistant = await client.assistants.create(graph_name, config={"configurable": {"model_name": "openai"}}, name="openai_assistant")
    ```

=== "Javascript"

    ```js
    const openaiAssistant = await client.assistants.create({graphId: graphName, config: { configurable: {"modelName": "openai"}}, name: "openaiAssistant"});
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPOLYMENT_URL>/assistants \
    --header 'Content-Type: application/json' \
    --data '{
    "graph_id": "agent",
    "config": {"model_name": "openai"},
    "name": "openai_assistant"
    }'
    ```

#### Using the studio

To create an assistant using the studio do the following steps:

1. Click on the "Create New Assistant" button:

    ![click create](./img/click_create_assistant.png)

1. Use the create assistant pane to enter info for the assistant you wish to create, and then click create:

    ![create](./img/create_assistant.png)

1. See that your assistant was created and is displayed in the Studio

    ![view create](./img/create_assistant_view.png)

1. Click on the edit button next to the selected assistant to manage your created assistant:

    ![create edit](./img/edit_created_assistant.png)

### Create a new version for your assistant

Let's now say we wanted to add a system prompt to our assistant. We can do this by using the `update` endpoint as follows. Please note that you must pass in the ENTIRE config (and metadata if you are using it). The update endpoint creates new versions completely from scratch and does not rely on previously entered config. In this case, we need to continue telling the assistant to use "openai" as the model.

=== "Python"

    ```python
    openai_assistant_v2 = await client.assistants.update(openai_assistant['assistant_id'], config={"configurable": {"model_name": "openai", "system_prompt": "You are a helpful assistant!"}})
    ```

=== "Javascript"

    ```js
    const openaiAssistantV2 = await client.assistants.update(openaiAssistant['assistant_id'], {config: { configurable: {"modelName": "openai", "systemPrompt": "You are a helpful assistant!"}}});
    ```

=== "CURL"

    ```bash
    curl --request PATCH \
    --url <DEPOLYMENT_URL>/assistants/<ASSISTANT_ID> \
    --header 'Content-Type: application/json' \
    --data '{
    "config": {"model_name": "openai", "system_prompt": "You are a helpful assistant!"}
    }'
    ```

#### Using the studio

1. First, click on the edit button next to the `openai_assistant`. Then, add a system prompt and click "Save New Version":

    ![create new version](./img/create_new_version.png)

1. Then you can see it is selected in the assistant dropdown:

    ![see version dropdown](./img/see_new_version.png)

1. And you can see all the version history in the edit pane for the assistant:

    ![see versions](./img/see_version_history.png)

### Point your assistant to a different version

After having created multiple versions, we can change the version our assistant points to both by using the SDK and also the Studio. In this case we will be resetting the `openai_assistant` we just created two versions for to point back to the first version. When you create a new version (by using the `update` endpoint) the assistant automatically points to the newly created version, so following the code above our `openai_assistant` is pointing to the second version. Here we will change it to point to the first version:

=== "Python"

    ```python
    await client.assistants.set_latest(openai_assistant['assistant_id'], 1)
    ```

=== "Javascript"

    ```js
    await client.assistants.setLatest(openaiAssistant['assistant_id'], 1);
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/assistants/<ASSISTANT_ID>/latest \
    --header 'Content-Type: application/json' \
    --data '{
    "version": 1
    }'
    ```


#### Using the studio

To change the version, all you have to do is click into the edit pane for an assistant, select the version you want to change to, and then click the "Set As Current Version" button

![set version](./img/select_different_version.png)

### Using your assistant versions

Whether you are a business user iterating without writing code, or a developer using the SDK - assistant versioning allows you to quickly test different agents in a controlled environment, making it easy to iterate fast. You can use any of the assistant versions just how you would a normal assistant, and can read more about how to stream output from these assistants by reading [these guides](https://langchain-ai.github.io/langgraph/cloud/how-tos/#streaming) or [this one](https://langchain-ai.github.io/langgraph/cloud/how-tos/invoke_studio/) if you are using the Studio.

!!! warning "Deleting Assistants"
    Deleting as assistant will delete ALL of it's versions, since they all point to the same assistant ID. There is currently no way to just delete a single version, but by pointing your assistant to the correct version you can skip any versions that you don't wish to use.

<a name="docscloudhow-tosbackground_runmd"></a>

## How to kick off background runs
 

This guide covers how to kick off background runs for your agent.
This can be useful for long running jobs.

### Setup

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

Output:

    {
        'thread_id': '5cb1e8a1-34b3-4a61-a34e-71a9799bd00d',
        'created_at': '2024-08-30T20:35:52.062934+00:00',
        'updated_at': '2024-08-30T20:35:52.062934+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

### Check runs on thread

If we list the current runs on this thread, we will see that it's empty:

=== "Python"

    ```python
    runs = await client.runs.list(thread["thread_id"])
    print(runs)
    ```

=== "Javascript"

    ```js
    let runs = await client.runs.list(thread['thread_id']);
    console.log(runs);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs
    ```

Output:

    []

### Start runs on thread

Now let's kick off a run:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    run = await client.runs.create(thread["thread_id"], assistant_id, input=input)
    ```

=== "Javascript"

    ```js
    let input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]};
    let run = await client.runs.create(thread["thread_id"], assistantID, { input });
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>
        }'
    ```

The first time we poll it, we can see `status=pending`:

=== "Python"

    ```python
    print(await client.runs.get(thread["thread_id"], run["run_id"]))
    ```

=== "Javascript"

    ```js
    console.log(await client.runs.get(thread["thread_id"], run["run_id"]));
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>
    ```

Output:

        {
            "run_id": "1ef6a5f8-bd86-6763-bbd6-bff042db7b1b",
            "thread_id": "7885f0cf-94ad-4040-91d7-73f7ba007c8a",
            "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
            "created_at": "2024-09-04T01:46:47.244887+00:00",
            "updated_at": "2024-09-04T01:46:47.244887+00:00",
            "metadata": {},
            "status": "pending",
            "kwargs": {
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "what's the weather in sf"
                        }
                    ]
                },
                "config": {
                    "metadata": {
                        "created_by": "system"
                    },
                    "configurable": {
                        "run_id": "1ef6a5f8-bd86-6763-bbd6-bff042db7b1b",
                        "user_id": "",
                        "graph_id": "agent",
                        "thread_id": "7885f0cf-94ad-4040-91d7-73f7ba007c8a",
                        "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
                        "checkpoint_id": null
                    }
                },
                "webhook": null,
                "temporary": false,
                "stream_mode": [
                    "values"
                ],
                "feedback_keys": null,
                "interrupt_after": null,
                "interrupt_before": null
            },
            "multitask_strategy": "reject"
        }



Now we can join the run, wait for it to finish and check that status again:

=== "Python"

    ```python
    await client.runs.join(thread["thread_id"], run["run_id"])
    print(await client.runs.get(thread["thread_id"], run["run_id"]))
    ```

=== "Javascript"

    ```js
    await client.runs.join(thread["thread_id"], run["run_id"]);
    console.log(await client.runs.get(thread["thread_id"], run["run_id"]));
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/join &&
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>
    ```

Output:

    {
        "run_id": "1ef6a5f8-bd86-6763-bbd6-bff042db7b1b",
        "thread_id": "7885f0cf-94ad-4040-91d7-73f7ba007c8a",
        "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
        "created_at": "2024-09-04T01:46:47.244887+00:00",
        "updated_at": "2024-09-04T01:46:47.244887+00:00",
        "metadata": {},
        "status": "success",
        "kwargs": {
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "what's the weather in sf"
                    }
                ]
            },
            "config": {
                "metadata": {
                    "created_by": "system"
                },
                "configurable": {
                    "run_id": "1ef6a5f8-bd86-6763-bbd6-bff042db7b1b",
                    "user_id": "",
                    "graph_id": "agent",
                    "thread_id": "7885f0cf-94ad-4040-91d7-73f7ba007c8a",
                    "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
                    "checkpoint_id": null
                }
            },
            "webhook": null,
            "temporary": false,
            "stream_mode": [
                "values"
            ],
            "feedback_keys": null,
            "interrupt_after": null,
            "interrupt_before": null
        },
        "multitask_strategy": "reject"
    }


Perfect! The run succeeded as we would expect. We can double check that the run worked as expected by printing out the final state:

=== "Python"

    ```python
    final_result = await client.threads.get_state(thread["thread_id"])
    print(final_result)
    ```

=== "Javascript"

    ```js
    let finalResult = await client.threads.getState(thread["thread_id"]);
    console.log(finalResult);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state
    ```

Output:

    {
        "values": {
            "messages": [
                {
                    "content": "what's the weather in sf",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "human",
                    "name": null,
                    "id": "beba31bf-320d-4125-9c37-cadf526ac47a",
                    "example": false
                },
                {
                    "content": [
                        {
                            "id": "toolu_01AaNPSPzqia21v7aAKwbKYm",
                            "input": {},
                            "name": "tavily_search_results_json",
                            "type": "tool_use",
                            "index": 0,
                            "partial_json": "{\"query\": \"weather in san francisco\"}"
                        }
                    ],
                    "additional_kwargs": {},
                    "response_metadata": {
                        "stop_reason": "tool_use",
                        "stop_sequence": null
                    },
                    "type": "ai",
                    "name": null,
                    "id": "run-f220faf8-1d27-4f73-ad91-6bb3f47e8639",
                    "example": false,
                    "tool_calls": [
                        {
                            "name": "tavily_search_results_json",
                            "args": {
                                "query": "weather in san francisco"
                            },
                            "id": "toolu_01AaNPSPzqia21v7aAKwbKYm",
                            "type": "tool_call"
                        }
                    ],
                    "invalid_tool_calls": [],
                    "usage_metadata": {
                        "input_tokens": 273,
                        "output_tokens": 61,
                        "total_tokens": 334
                    }
                },
                {
                    "content": "[{\"url\": \"https://www.weatherapi.com/\", \"content\": \"{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1725052131, 'localtime': '2024-08-30 14:08'}, 'current': {'last_updated_epoch': 1725051600, 'last_updated': '2024-08-30 14:00', 'temp_c': 21.1, 'temp_f': 70.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 11.9, 'wind_kph': 19.1, 'wind_degree': 290, 'wind_dir': 'WNW', 'pressure_mb': 1018.0, 'pressure_in': 30.07, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 59, 'cloud': 25, 'feelslike_c': 21.1, 'feelslike_f': 70.0, 'windchill_c': 18.6, 'windchill_f': 65.5, 'heatindex_c': 18.6, 'heatindex_f': 65.5, 'dewpoint_c': 12.2, 'dewpoint_f': 54.0, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 15.0, 'gust_kph': 24.2}}\"}]",
                    "additional_kwargs": {},
                    "response_metadata": {},
                    "type": "tool",
                    "name": "tavily_search_results_json",
                    "id": "686b2487-f332-4e58-9508-89b3a814cd81",
                    "tool_call_id": "toolu_01AaNPSPzqia21v7aAKwbKYm",
                    "artifact": {
                        "query": "weather in san francisco",
                        "follow_up_questions": null,
                        "answer": null,
                        "images": [],
                        "results": [
                            {
                                "title": "Weather in San Francisco",
                                "url": "https://www.weatherapi.com/",
                                "content": "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1725052131, 'localtime': '2024-08-30 14:08'}, 'current': {'last_updated_epoch': 1725051600, 'last_updated': '2024-08-30 14:00', 'temp_c': 21.1, 'temp_f': 70.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 11.9, 'wind_kph': 19.1, 'wind_degree': 290, 'wind_dir': 'WNW', 'pressure_mb': 1018.0, 'pressure_in': 30.07, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 59, 'cloud': 25, 'feelslike_c': 21.1, 'feelslike_f': 70.0, 'windchill_c': 18.6, 'windchill_f': 65.5, 'heatindex_c': 18.6, 'heatindex_f': 65.5, 'dewpoint_c': 12.2, 'dewpoint_f': 54.0, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 15.0, 'gust_kph': 24.2}}",
                                "score": 0.976148,
                                "raw_content": null
                            }
                        ],
                        "response_time": 3.07
                    },
                    "status": "success"
                },
                {
                    "content": [
                        {
                            "text": "\n\nThe search results provide the current weather conditions in San Francisco. According to the data, as of 2:00 PM on August 30, 2024, the temperature in San Francisco is 70\u00b0F (21.1\u00b0C) with partly cloudy skies. The wind is blowing from the west-northwest at around 12 mph (19 km/h). The humidity is 59% and visibility is 9 miles (16 km). Overall, it looks like a nice late summer day in San Francisco with comfortable temperatures and partly sunny conditions.",
                            "type": "text",
                            "index": 0
                        }
                    ],
                    "additional_kwargs": {},
                    "response_metadata": {
                        "stop_reason": "end_turn",
                        "stop_sequence": null
                    },
                    "type": "ai",
                    "name": null,
                    "id": "run-8fecc61d-3d9f-4e16-8e8a-92f702be498a",
                    "example": false,
                    "tool_calls": [],
                    "invalid_tool_calls": [],
                    "usage_metadata": {
                        "input_tokens": 837,
                        "output_tokens": 124,
                        "total_tokens": 961
                    }
                }
            ]
        },
        "next": [],
        "tasks": [],
        "metadata": {
            "step": 3,
            "run_id": "1ef67140-eb23-684b-8253-91d4c90bb05e",
            "source": "loop",
            "writes": {
                "agent": {
                    "messages": [
                        {
                            "id": "run-8fecc61d-3d9f-4e16-8e8a-92f702be498a",
                            "name": null,
                            "type": "ai",
                            "content": [
                                {
                                    "text": "\n\nThe search results provide the current weather conditions in San Francisco. According to the data, as of 2:00 PM on August 30, 2024, the temperature in San Francisco is 70\u00b0F (21.1\u00b0C) with partly cloudy skies. The wind is blowing from the west-northwest at around 12 mph (19 km/h). The humidity is 59% and visibility is 9 miles (16 km). Overall, it looks like a nice late summer day in San Francisco with comfortable temperatures and partly sunny conditions.",
                                    "type": "text",
                                    "index": 0
                                }
                            ],
                            "example": false,
                            "tool_calls": [],
                            "usage_metadata": {
                                "input_tokens": 837,
                                "total_tokens": 961,
                                "output_tokens": 124
                            },
                            "additional_kwargs": {},
                            "response_metadata": {
                                "stop_reason": "end_turn",
                                "stop_sequence": null
                            },
                            "invalid_tool_calls": []
                        }
                    ]
                }
            },
            "user_id": "",
            "graph_id": "agent",
            "thread_id": "5cb1e8a1-34b3-4a61-a34e-71a9799bd00d",
            "created_by": "system",
            "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca"
        },
        "created_at": "2024-08-30T21:09:00.079909+00:00",
        "checkpoint_id": "1ef67141-3ca2-6fae-8003-fe96832e57d6",
        "parent_checkpoint_id": "1ef67141-2129-6b37-8002-61fc3bf69cb5"
    }

We can also just print the content of the last AIMessage:

=== "Python"

    ```python
    print(final_result['values']['messages'][-1]['content'][0]['text'])
    ```

=== "Javascript"

    ```js
    console.log(finalResult['values']['messages'][finalResult['values']['messages'].length-1]['content'][0]['text']);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | jq -r '.values.messages[-1].content.[0].text'
    ```

Output:

    The search results provide the current weather conditions in San Francisco. According to the data, as of 2:00 PM on August 30, 2024, the temperature in San Francisco is 70┬░F (21.1┬░C) with partly cloudy skies. The wind is blowing from the west-northwest at around 12 mph (19 km/h). The humidity is 59% and visibility is 9 miles (16 km). Overall, it looks like a nice late summer day in San Francisco with comfortable temperatures and partly sunny conditions.

<a name="docscloudhow-toscheck_thread_statusmd"></a>

## Check the Status of your Threads

### Setup

To start, we can setup our client with whatever URL you are hosting your graph from:

#### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Find idle threads

We can use the following commands to find threads that are idle, which means that all runs executed on the thread have finished running:

=== "Python"

    ```python
    print(await client.threads.search(status="idle",limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({ status: "idle", limit: 1 }));
    ```

=== "CURL"

    ```bash
    curl --request POST \  
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"status": "idle", "limit": 1}'
    ```

Output:

    [{'thread_id': 'cacf79bb-4248-4d01-aabc-938dbd60ed2c',
    'created_at': '2024-08-14T17:36:38.921660+00:00',
    'updated_at': '2024-08-14T17:36:38.921660+00:00',
    'metadata': {'graph_id': 'agent'},
    'status': 'idle',
    'config': {'configurable': {}}}]


### Find interrupted threads

We can use the following commands to find threads that have been interrupted in the middle of a run, which could either mean an error occurred before the run finished or a human-in-the-loop breakpoint was reached and the run is waiting to continue: 

=== "Python"

    ```python
    print(await client.threads.search(status="interrupted",limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({ status: "interrupted", limit: 1 }));
    ```

=== "CURL"

    ```bash
    curl --request POST \  
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"status": "interrupted", "limit": 1}'
    ```

Output:

    [{'thread_id': '0d282b22-bbd5-4d95-9c61-04dcc2e302a5',
    'created_at': '2024-08-14T17:41:50.235455+00:00',
    'updated_at': '2024-08-14T17:41:50.235455+00:00',
    'metadata': {'graph_id': 'agent'},
    'status': 'interrupted',
    'config': {'configurable': {}}}]
    
### Find busy threads

We can use the following commands to find threads that are busy, meaning they are currently handling the execution of a run:

=== "Python"

    ```python
    print(await client.threads.search(status="busy",limit=1))
    ```

=== "Javascript"

    ```js
    console.log(await client.threads.search({ status: "busy", limit: 1 }));
    ```

=== "CURL"

    ```bash
    curl --request POST \  
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"status": "busy", "limit": 1}'
    ```

Output:

    [{'thread_id': '0d282b22-bbd5-4d95-9c61-04dcc2e302a5',
    'created_at': '2024-08-14T17:41:50.235455+00:00',
    'updated_at': '2024-08-14T17:41:50.235455+00:00',
    'metadata': {'graph_id': 'agent'},
    'status': 'busy',
    'config': {'configurable': {}}}]

### Find specific threads

You may also want to check the status of specific threads, which you can do in a few ways:

#### Find by ID

You can use the `get` function to find the status of a specific thread, as long as you have the ID saved

=== "Python"

    ```python
    print((await client.threads.get(<THREAD_ID>))['status'])
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.get(<THREAD_ID>)).status);
    ```

=== "CURL"

    ```bash
    curl --request GET \ 
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID> \
    --header 'Content-Type: application/json' | jq -r '.status'
    ```

Output:

    'idle'

#### Find by metadata

The search endpoint for threads also allows you to filter on metadata, which can be helpful if you use metadata to tag threads in order to keep them organized:

=== "Python"

    ```python
    print((await client.threads.search(metadata={"foo":"bar"},limit=1))[0]['status'])
    ```

=== "Javascript"

    ```js
    console.log((await client.threads.search({ metadata: { "foo": "bar" }, limit: 1 }))[0].status);
    ```

=== "CURL"

    ```bash
    curl --request POST \  
    --url <DEPLOYMENT_URL>/threads/search \
    --header 'Content-Type: application/json' \
    --data '{"metadata": {"foo":"bar"}, "limit": 1}' | jq -r '.[0].status'
    ```

Output:

    'idle'

<a name="docscloudhow-tosconfiguration_cloudmd"></a>

## How to create agents with configuration

One of the benefits of LangGraph API is that it lets you create agents with different configurations.
This is useful when you want to:

- Define a cognitive architecture once as a LangGraph
- Let that LangGraph be configurable across some attributes (for example, system message or LLM to use)
- Let users create agents with arbitrary configurations, save them, and then use them in the future

In this guide we will show how to do that for the default agent we have built in.

If you look at the agent we defined, you can see that inside the `call_model` node we have created the model based on some configuration. That node looks like:

=== "Python"

    ```python
    def call_model(state, config):
        messages = state["messages"]
        model_name = config.get('configurable', {}).get("model_name", "anthropic")
        model = _get_model(model_name)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}
    ```

=== "Javascript"

    ```js
    function callModel(state: State, config: RunnableConfig) {
      const messages = state.messages;
      const modelName = config.configurable?.model_name ?? "anthropic";
      const model = _getModel(modelName);
      const response = model.invoke(messages);
      // We return a list, because this will get added to the existing list
      return { messages: [response] };
    }
    ```

We are looking inside the config for a `model_name` parameter (which defaults to `anthropic` if none is found). That means that by default we are using Anthropic as our model provider. In this example we will see an example of how to create an example agent that is configured to use OpenAI.

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Select an assistant that is not configured
    assistants = await client.assistants.search()
    assistant = [a for a in assistants if not a["config"]][0]
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Select an assistant that is not configured
    const assistants = await client.assistants.search();
    const assistant = assistants.find(a => !a.config);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0]'
    ```

We can now call `.get_schemas` to get schemas associated with this graph:

=== "Python"

    ```python
    schemas = await client.assistants.get_schemas(
        assistant_id=assistant["assistant_id"]
    )
    # There are multiple types of schemas
    # We can get the `config_schema` to look at the configurable parameters
    print(schemas["config_schema"])
    ```

=== "Javascript"

    ```js
    const schemas = await client.assistants.getSchemas(
      assistant["assistant_id"]
    );
    // There are multiple types of schemas
    // We can get the `config_schema` to look at the configurable parameters
    console.log(schemas.config_schema);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/assistants/<ASSISTANT_ID>/schemas | jq -r '.config_schema'
    ```

Output:

    {
        'model_name': 
            {
                'title': 'Model Name',
                'enum': ['anthropic', 'openai'],
                'type': 'string'
            }
    }

Now we can initialize an assistant with config:

=== "Python"

    ```python
    openai_assistant = await client.assistants.create(
        # "agent" is the name of a graph we deployed
        "agent", config={"configurable": {"model_name": "openai"}}
    )

    print(openai_assistant)
    ```

=== "Javascript"

    ```js
    let openAIAssistant = await client.assistants.create(
      // "agent" is the name of a graph we deployed
      "agent", { "configurable": { "model_name": "openai" } }
    );

    console.log(openAIAssistant);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants \
        --header 'Content-Type: application/json' \
        --data '{"graph_id":"agent","config":{"configurable":{"model_name":"open_ai"}}}'
    ```

Output:

    {
        "assistant_id": "62e209ca-9154-432a-b9e9-2d75c7a9219b",
        "graph_id": "agent",
        "created_at": "2024-08-31T03:09:10.230718+00:00",
        "updated_at": "2024-08-31T03:09:10.230718+00:00",
        "config": {
            "configurable": {
                "model_name": "open_ai"
            }
        },
        "metadata": {}
    }

We can verify the config is indeed taking effect:

=== "Python"

    ```python
    thread = await client.threads.create()
    input = {"messages": [{"role": "user", "content": "who made you?"}]}
    async for event in client.runs.stream(
        thread["thread_id"],
        openai_assistant["assistant_id"],
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving event of type: {event.event}")
        print(event.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const thread = await client.threads.create();
    let input = { "messages": [{ "role": "user", "content": "who made you?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      openAIAssistant["assistant_id"],
      {
        input,
        streamMode: "updates"
      }
    );

    for await (const event of streamResponse) {
      console.log(`Receiving event of type: ${event.event}`);
      console.log(event.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    thread_id=$(curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}' | jq -r '.thread_id') && \
    curl --request POST \
        --url "<DEPLOYMENT_URL>/threads/${thread_id}/runs/stream" \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <OPENAI_ASSISTANT_ID>,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "who made you?"
                    }
                ]
            },
            "stream_mode": [
                "updates"
            ]
        }' | \
        sed 's/\r$//' | \
        awk '
        /^event:/ {
            if (data_content != "") {
                print data_content "\n"
            }
            sub(/^event: /, "Receiving event of type: ", $0)
            printf "%s...\n", $0
            data_content = ""
        }
        /^data:/ {
            sub(/^data: /, "", $0)
            data_content = $0
        }
        END {
            if (data_content != "") {
                print data_content "\n\n"
            }
        }
    '
    ```

Output:

    Receiving event of type: metadata
    {'run_id': '1ef6746e-5893-67b1-978a-0f1cd4060e16'}



    Receiving event of type: updates
    {'agent': {'messages': [{'content': 'I was created by OpenAI, a research organization focused on developing and advancing artificial intelligence technology.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-e1a6b25c-8416-41f2-9981-f9cfe043f414', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}





<a name="docscloudhow-toscopy_threadsmd"></a>

## Copying Threads

You may wish to copy (i.e. "fork") an existing thread in order to keep the existing thread's history and create independent runs that do not affect the original thread. This guide shows how you can do that.

### Setup

This code assumes you already have a thread to copy. You can read about what a thread is [here](#threads) and learn how to stream a run on a thread in [these how-to guides](#streaming_1).

#### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url="<DEPLOYMENT_URL>")
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: "<DEPLOYMENT_URL>" });
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{
        "metadata": {}
      }'
    ```

### Copying a thread

The code below assumes that a thread you'd like to copy already exists.

Copying a thread will create a new thread with the same history as the existing thread, and then allow you to continue executing runs.

#### Create copy

=== "Python"

    ```python
    copied_thread = await client.threads.copy(<THREAD_ID>)
    ```

=== "Javascript"

    ```js
    let copiedThread = await client.threads.copy(<THREAD_ID>);
    ```

=== "CURL"

    ```bash
    curl --request POST --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/copy \
    --header 'Content-Type: application/json'
    ```

#### Verify copy

We can verify that the history from the prior thread did indeed copy over correctly:

=== "Python"

    ```python
    def remove_thread_id(d):
      if 'metadata' in d and 'thread_id' in d['metadata']:
          del d['metadata']['thread_id']
      return d

    original_thread_history = list(map(remove_thread_id,await client.threads.get_history(<THREAD_ID>)))
    copied_thread_history = list(map(remove_thread_id,await client.threads.get_history(copied_thread['thread_id'])))

    # Compare the two histories
    assert original_thread_history == copied_thread_history
    # if we made it here the assertion passed!
    print("The histories are the same.")
    ```

=== "Javascript"

    ```js
    function removeThreadId(d) {
      if (d.metadata && d.metadata.thread_id) {
        delete d.metadata.thread_id;
      }
      return d;
    }

    // Assuming `client.threads.getHistory(threadId)` is an async function that returns a list of dicts
    async function compareThreadHistories(threadId, copiedThreadId) {
      const originalThreadHistory = (await client.threads.getHistory(threadId)).map(removeThreadId);
      const copiedThreadHistory = (await client.threads.getHistory(copiedThreadId)).map(removeThreadId);

      // Compare the two histories
      console.assert(JSON.stringify(originalThreadHistory) === JSON.stringify(copiedThreadHistory));
      // if we made it here the assertion passed!
      console.log("The histories are the same.");
    }

    // Example usage
    compareThreadHistories(<THREAD_ID>, copiedThread.thread_id);
    ```

=== "CURL"

    ```bash
    if diff <(
        curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/history | jq -S 'map(del(.metadata.thread_id))'
    ) <(
        curl --request GET --url <DEPLOYMENT_URL>/threads/<COPIED_THREAD_ID>/history | jq -S 'map(del(.metadata.thread_id))'
    ) >/dev/null; then
        echo "The histories are the same."
    else
        echo "The histories are different."
    fi
    ```

Output:

    The histories are the same.

<a name="docscloudhow-toscron_jobsmd"></a>

## Cron Jobs

Sometimes you don't want to run your graph based on user interaction, but rather you would like to schedule your graph to run on a schedule - for example if you wish for your graph to compose and send out a weekly email of to-dos for your team. LangGraph Cloud allows you to do this without having to write your own script by using the `Crons` client. To schedule a graph job, you need to pass a [cron expression](https://crontab.cronhub.io/) to inform the client when you want to run the graph. `Cron` jobs are run in the background and do not interfere with normal invocations of the graph.

### Setup

First, let's setup our SDK client, assistant, and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0].graph_id' && \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

Output:

    {
        'thread_id': '9dde5490-2b67-47c8-aa14-4bfec88af217', 
        'created_at': '2024-08-30T23:07:38.242730+00:00', 
        'updated_at': '2024-08-30T23:07:38.242730+00:00', 
        'metadata': {}, 
        'status': 'idle', 
        'config': {}, 
        'values': None
    }

### Cron job on a thread 

To create a cron job associated with a specific thread, you can write:


=== "Python"

    ```python
    # This schedules a job to run at 15:27 (3:27PM) every day
    cron_job = await client.crons.create_for_thread(
        thread["thread_id"],
        assistant_id,
        schedule="27 15 * * *",
        input={"messages": [{"role": "user", "content": "What time is it?"}]},
    )
    ```

=== "Javascript"

    ```js
    // This schedules a job to run at 15:27 (3:27PM) every day
    const cronJob = await client.crons.create_for_thread(
      thread["thread_id"],
      assistantId,
      {
        schedule: "27 15 * * *",
        input: { messages: [{ role: "user", content: "What time is it?" }] }
      }
    );
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/crons \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>,
        }'
    ```

Note that it is **very** important to delete `Cron` jobs that are no longer useful. Otherwise you could rack up unwanted API charges to the LLM! You can delete a `Cron` job using the following code:

=== "Python"

    ```python
    await client.crons.delete(cron_job["cron_id"])
    ```

=== "Javascript"

    ```js
    await client.crons.delete(cronJob["cron_id"]);
    ```

=== "CURL"

    ```bash
    curl --request DELETE \
        --url <DEPLOYMENT_URL>/runs/crons/<CRON_ID>
    ```

### Cron job stateless

You can also create stateless cron jobs by using the following code:

=== "Python"

    ```python
    # This schedules a job to run at 15:27 (3:27PM) every day
    cron_job_stateless = await client.crons.create(
        assistant_id,
        schedule="27 15 * * *",
        input={"messages": [{"role": "user", "content": "What time is it?"}]},
    )
    ```

=== "Javascript"

    ```js
    // This schedules a job to run at 15:27 (3:27PM) every day
    const cronJobStateless = await client.crons.create(
      assistantId,
      {
        schedule: "27 15 * * *",
        input: { messages: [{ role: "user", content: "What time is it?" }] }
      }
    );
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/runs/crons \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>,
        }'
    ```

Again, remember to delete your job once you are done with it!

=== "Python"

    ```python
    await client.crons.delete(cron_job_stateless["cron_id"])
    ```

=== "Javascript"

    ```js
    await client.crons.delete(cronJobStateless["cron_id"]);
    ```

=== "CURL"

    ```bash
    curl --request DELETE \
        --url <DEPLOYMENT_URL>/runs/crons/<CRON_ID>
    ```


<a name="docscloudhow-tosdatasets_studiomd"></a>

## Adding nodes as dataset examples in Studio

In LangGraph Studio you can create dataset examples from the thread history in the right-hand pane. This can be especially useful when you want to evaluate intermediate steps of the agent.

1. Click on the `Add to Dataset` button to enter the dataset mode.
1. Select nodes which you want to add to dataset.
1. Select the target dataset to create the example in.

You can edit the example payload before sending it to the dataset, which is useful if you need to make changes to conform the example to the dataset schema.

Finally, you can customise the target dataset by clicking on the `Settings` button.

See [Evaluating intermediate steps](https://docs.smith.langchain.com/evaluation/how_to_guides/langgraph#evaluating-intermediate-steps) for more details on how to evaluate intermediate steps.

<video controls allowfullscreen="true" poster="../img/studio_datasets.jpg">
    <source src="https://langgraph-docs-assets.pages.dev/studio_datasets.mp4" type="video/mp4">
</video>


<a name="docscloudhow-tosenqueue_concurrentmd"></a>

## Enqueue

This guide assumes knowledge of what double-texting is, which you can learn about in the [double-texting conceptual guide](#docsconceptsdouble_textingmd).

The guide covers the `enqueue` option for double texting, which adds the interruptions to a queue and executes them in the order they are received by the client. Below is a quick example of using the `enqueue` option.

### Setup

First, we will define a quick helper function for printing out JS and CURL model outputs (you can skip this if using Python):

=== "Javascript"

    ```js
    function prettyPrint(m) {
      const padded = " " + m['type'] + " ";
      const sepLen = Math.floor((80 - padded.length) / 2);
      const sep = "=".repeat(sepLen);
      const secondSep = sep + (padded.length % 2 ? "=" : "");
      
      console.log(`${sep}${padded}${secondSep}`);
      console.log("\n\n");
      console.log(m.content);
    }
    ```

=== "CURL"

    ```bash
    # PLACE THIS IN A FILE CALLED pretty_print.sh
    pretty_print() {
      local type="$1"
      local content="$2"
      local padded=" $type "
      local total_width=80
      local sep_len=$(( (total_width - ${#padded}) / 2 ))
      local sep=$(printf '=%.0s' $(eval "echo {1.."${sep_len}"}"))
      local second_sep=$sep
      if (( (total_width - ${#padded}) % 2 )); then
        second_sep="${second_sep}="
      fi

      echo "${sep}${padded}${second_sep}"
      echo
      echo "$content"
    }
    ```

Then, let's import our required packages and instantiate our client, assistant, and thread.

=== "Python"

    ```python
    import asyncio

    import httpx
    from langchain_core.messages import convert_to_messages
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";
    

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```
  
=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Create runs

Now let's start two runs, with the second interrupting the first one with a multitask strategy of "enqueue":

=== "Python"

    ```python
    first_run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    )
    second_run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": "what's the weather in nyc?"}]},
        multitask_strategy="enqueue",
    )
    ```

=== "Javascript"

    ```js
    const firstRun = await client.runs.create(
      thread["thread_id"],
      assistantId,
      input={"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    )

    const secondRun = await client.runs.create(
      thread["thread_id"],
      assistantId,
      input={"messages": [{"role": "user", "content": "what's the weather in nyc?"}]},
      multitask_strategy="enqueue",
    )
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in sf?\"}]},
    }" && curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in nyc?\"}]},
      \"multitask_strategy\": \"enqueue\"
    }"
    ```

### View run results

Verify that the thread has data from both runs:

=== "Python"

    ```python
    # wait until the second run completes
    await client.runs.join(thread["thread_id"], second_run["run_id"])

    state = await client.threads.get_state(thread["thread_id"])

    for m in convert_to_messages(state["values"]["messages"]):
        m.pretty_print()
    ```

=== "Javascript"

    ```js
    await client.runs.join(thread["thread_id"], secondRun["run_id"]);

    const state = await client.threads.getState(thread["thread_id"]);

    for (const m of state["values"]["messages"]) {
      prettyPrint(m);
    }
    ```

=== "CURL"

    ```bash
    source pretty_print.sh && curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/join && \
    curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | \
    jq -c '.values.messages[]' | while read -r element; do
        type=$(echo "$element" | jq -r '.type')
        content=$(echo "$element" | jq -r '.content | if type == "array" then tostring else . end')
        pretty_print "$type" "$content"
    done
    ```

Output:

    ================================ Human Message =================================
    
    what's the weather in sf?
    ================================== Ai Message ==================================
    
    [{'id': 'toolu_01Dez1sJre4oA2Y7NsKJV6VT', 'input': {'query': 'weather in san francisco'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01Dez1sJre4oA2Y7NsKJV6VT)
     Call ID: toolu_01Dez1sJre4oA2Y7NsKJV6VT
      Args:
        query: weather in san francisco
    ================================= Tool Message =================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629", "content": "Get the current and future weather conditions for San Francisco, CA, including temperature, precipitation, wind, air quality and more. See the hourly and 10-day outlook, radar maps, alerts and allergy information."}]
    ================================== Ai Message ==================================
    
    According to AccuWeather, the current weather conditions in San Francisco are:
    
    Temperature: 57┬░F (14┬░C)
    Conditions: Mostly Sunny
    Wind: WSW 10 mph
    Humidity: 72%
    
    The forecast for the next few days shows partly sunny skies with highs in the upper 50s to mid 60s F (14-18┬░C) and lows in the upper 40s to low 50s F (9-11┬░C). Typical mild, dry weather for San Francisco this time of year.
    
    Some key details from the AccuWeather forecast:
    
    Today: Mostly sunny, high of 62┬░F (17┬░C)
    Tonight: Partly cloudy, low of 49┬░F (9┬░C) 
    Tomorrow: Partly sunny, high of 59┬░F (15┬░C)
    Saturday: Mostly sunny, high of 64┬░F (18┬░C)
    Sunday: Partly sunny, high of 61┬░F (16┬░C)
    
    So in summary, expect seasonable spring weather in San Francisco over the next several days, with a mix of sun and clouds and temperatures ranging from the upper 40s at night to the low 60s during the days. Typical dry conditions with no rain in the forecast.
    ================================ Human Message =================================
    
    what's the weather in nyc?
    ================================== Ai Message ==================================
    
    [{'text': 'Here are the current weather conditions and forecast for New York City:', 'type': 'text'}, {'id': 'toolu_01FFft5Sx9oS6AdVJuRWWcGp', 'input': {'query': 'weather in new york city'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01FFft5Sx9oS6AdVJuRWWcGp)
     Call ID: toolu_01FFft5Sx9oS6AdVJuRWWcGp
      Args:
        query: weather in new york city
    ================================= Tool Message =================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'New York', 'region': 'New York', 'country': 'United States of America', 'lat': 40.71, 'lon': -74.01, 'tz_id': 'America/New_York', 'localtime_epoch': 1718734479, 'localtime': '2024-06-18 14:14'}, 'current': {'last_updated_epoch': 1718733600, 'last_updated': '2024-06-18 14:00', 'temp_c': 29.4, 'temp_f': 84.9, 'is_day': 1, 'condition': {'text': 'Sunny', 'icon': '//cdn.weatherapi.com/weather/64x64/day/113.png', 'code': 1000}, 'wind_mph': 2.2, 'wind_kph': 3.6, 'wind_degree': 158, 'wind_dir': 'SSE', 'pressure_mb': 1025.0, 'pressure_in': 30.26, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 63, 'cloud': 0, 'feelslike_c': 31.3, 'feelslike_f': 88.3, 'windchill_c': 28.3, 'windchill_f': 82.9, 'heatindex_c': 29.6, 'heatindex_f': 85.3, 'dewpoint_c': 18.4, 'dewpoint_f': 65.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 7.0, 'gust_mph': 16.5, 'gust_kph': 26.5}}"}]
    ================================== Ai Message ==================================
    
    According to the weather data from WeatherAPI:
    
    Current Conditions in New York City (as of 2:00 PM local time):
    - Temperature: 85┬░F (29┬░C)
    - Conditions: Sunny
    - Wind: 2 mph (4 km/h) from the SSE
    - Humidity: 63%
    - Heat Index: 85┬░F (30┬░C)
    
    The forecast shows sunny and warm conditions persisting over the next few days:
    
    Today: Sunny, high of 85┬░F (29┬░C)
    Tonight: Clear, low of 68┬░F (20┬░C)
    Tomorrow: Sunny, high of 88┬░F (31┬░C) 
    Thursday: Mostly sunny, high of 90┬░F (32┬░C)
    Friday: Partly cloudy, high of 87┬░F (31┬░C)
    
    So New York City is experiencing beautiful sunny weather with seasonably warm temperatures in the mid-to-upper 80s Fahrenheit (around 30┬░C). Humidity is moderate in the 60% range. Overall, ideal late spring/early summer conditions for being outdoors in the city over the next several days.



<a name="docscloudhow-toshuman_in_the_loop_breakpointmd"></a>

## How to Add Breakpoints

When creating LangGraph agents, it is often nice to add a human-in-the-loop component.
This can be helpful when giving them access to tools.
Often in these situations you may want to manually approve an action before taking.

This can be in several ways, but the primary supported way is to add an "interrupt" before a node is executed.
This interrupts execution at that node.
You can then resume from that spot to continue.  

### Setup

#### Code for your graph

In this how-to we use a simple ReAct style hosted graph (you can see the full code for defining it [here](../../how-tos/human_in_the_loop/breakpoints.ipynb)). The important thing is that there are two nodes (one named `agent` that calls the LLM, and one named `action` that calls the tool), and a routing function from `agent` that determines whether to call `action` next or just end the graph run (the `action` node always calls the `agent` node after execution).

#### SDK Initialization


=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Adding a breakpoint

We now want to add a breakpoint in our graph run, which we will do before a tool is called.
We can do this by adding `interrupt_before=["action"]`, which tells us to interrupt before calling the action node.
We can do this either when compiling the graph or when kicking off a run.
Here we will do it when kicking of a run, if you would like to to do it at compile time you need to edit the python file where your graph is defined and add the `interrupt_before` parameter when you call `.compile`.

First let's access our hosted LangGraph instance through the SDK:

And, now let's compile it with a breakpoint before the tool node:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
        interrupt_before=["action"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```
=== "Javascript"

    ```js
    const input = { messages: [{ role: "human", content: "what's the weather in sf" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
        interruptBefore: ["action"]
      }
    );

    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```
    
=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf\"}]},
       \"interrupt_before\": [\"action\"],
       \"stream_mode\": [
         \"messages\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "") {
             print data_content "\n"
         }
         sub(/^event: /, "Receiving event of type: ", $0)
         printf "%s...\n", $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    Receiving new event of type: metadata...
    {'run_id': '3b77ef83-687a-4840-8858-0371f91a92c3'}
    
    
    
    Receiving new event of type: data...
    {'agent': {'messages': [{'content': [{'id': 'toolu_01HwZqM1ptX6E15A5LAmyZTB', 'input': {'query': 'weather in san francisco'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-e5d17791-4d37-4ad2-815f-a0c4cba62585', 'example': False, 'tool_calls': [{'name': 'tavily_search_results_json', 'args': {'query': 'weather in san francisco'}, 'id': 'toolu_01HwZqM1ptX6E15A5LAmyZTB'}], 'invalid_tool_calls': []}]}}
    
    
    
    Receiving new event of type: end...
    None
    
    
    



<a name="docscloudhow-toshuman_in_the_loop_edit_statemd"></a>

## How to Edit State of a Deployed Graph

When creating LangGraph agents, it is often nice to add a human-in-the-loop component. This can be helpful when giving them access to tools. Often in these situations you may want to edit the graph state before continuing (for example, to edit what tool is being called, or how it is being called).

This can be in several ways, but the primary supported way is to add an "interrupt" before a node is executed. This interrupts execution at that node. You can then use update_state to update the state, and then resume from that spot to continue.

### Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](#agent) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input.

#### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:


=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Editing state

#### Initial invocation

Now let's invoke our graph, making sure to interrupt before the `action` node.

=== "Python"

    ```python
    input = { 'messages':[{ "role":"user", "content":"search for weather in SF" }] }

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
        interrupt_before=["action"],
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { messages: [{ role: "human", content: "search for weather in SF" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
        interruptBefore: ["action"],
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"search for weather in SF\"}]},
       \"interrupt_before\": [\"action\"],
       \"stream_mode\": [
         \"updates\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'agent': {'messages': [{'content': [{'text': "Certainly! I'll search for the current weather in San Francisco for you using the search function. Here's how I'll do that:", 'type': 'text'}, {'id': 'toolu_01KEJMBFozSiZoS4mAcPZeqQ', 'input': {'query': 'current weather in San Francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-6dbb0167-f8f6-4e2a-ab68-229b2d1fbb64', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in San Francisco'}, 'id': 'toolu_01KEJMBFozSiZoS4mAcPZeqQ'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


#### Edit the state

Now, let's assume we actually meant to search for the weather in Sidi Frej (another city with the initials SF). We can edit the state to properly reflect that:


=== "Python"

    ```python
    # First, lets get the current state
    current_state = await client.threads.get_state(thread['thread_id'])

    # Let's now get the last message in the state
    # This is the one with the tool calls that we want to update
    last_message = current_state['values']['messages'][-1]

    # Let's now update the args for that tool call
    last_message['tool_calls'][0]['args'] = {'query': 'current weather in Sidi Frej'}

    # Let's now call `update_state` to pass in this message in the `messages` key
    # This will get treated as any other update to the state
    # It will get passed to the reducer function for the `messages` key
    # That reducer function will use the ID of the message to update it
    # It's important that it has the right ID! Otherwise it would get appended
    # as a new message
    await client.threads.update_state(thread['thread_id'], {"messages": last_message})
    ```

=== "Javascript"

    ```js
    // First, let's get the current state
    const currentState = await client.threads.getState(thread["thread_id"]);

    // Let's now get the last message in the state
    // This is the one with the tool calls that we want to update
    let lastMessage = currentState.values.messages.slice(-1)[0];

    // Let's now update the args for that tool call
    lastMessage.tool_calls[0].args = { query: "current weather in Sidi Frej" };

    // Let's now call `update_state` to pass in this message in the `messages` key
    // This will get treated as any other update to the state
    // It will get passed to the reducer function for the `messages` key
    // That reducer function will use the ID of the message to update it
    // It's important that it has the right ID! Otherwise it would get appended
    // as a new message
    await client.threads.updateState(thread["thread_id"], { values: { messages: lastMessage } });
    ```

=== "CURL"

    ```bash
    curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | \                                                                                      
    jq '.values.messages[-1] | (.tool_calls[0].args = {"query": "current weather in Sidi Frej"})' | \
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
      --header 'Content-Type: application/json' \
      --data @-
    ```

Output:

    {'configurable': {'thread_id': '9c8f1a43-9dd8-4017-9271-2c53e57cf66a',
      'checkpoint_ns': '',
      'checkpoint_id': '1ef58e7e-3641-649f-8002-8b4305a64858'}}



#### Resume invocation

Now we can resume our graph run but with the updated state:


=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
        stream_mode="updates",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```
=== "Javascript"

    ```js
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
        streamMode: "updates",
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \                                                                             
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"stream_mode\": [
         \"updates\"
       ]
     }"| \ 
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'action': {'messages': [{'content': '["I looked up: current weather in Sidi Frej. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini ­ƒÿê."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '1161b8d1-bee4-4188-9be8-698aecb69f10', 'tool_call_id': 'toolu_01KEJMBFozSiZoS4mAcPZeqQ'}]}}
    {'agent': {'messages': [{'content': [{'text': 'I apologize for the confusion in my search query. It seems the search function interpreted "SF" as "Sidi Frej" instead of "San Francisco" as we intended. Let me search again with the full city name to get the correct information:', 'type': 'text'}, {'id': 'toolu_0111rrwgfAcmurHZn55qjqTR', 'input': {'query': 'current weather in San Francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-b8c25779-cfb4-46fc-a421-48553551242f', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in San Francisco'}, 'id': 'toolu_0111rrwgfAcmurHZn55qjqTR'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}
    {'action': {'messages': [{'content': '["I looked up: current weather in San Francisco. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini ­ƒÿê."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '6bc632ae-5ee6-4d01-9532-79c524a2d443', 'tool_call_id': 'toolu_0111rrwgfAcmurHZn55qjqTR'}]}}
    {'agent': {'messages': [{'content': "Now, based on the search results, I can provide you with information about the current weather in San Francisco:\n\nThe weather in San Francisco is currently sunny. \n\nIt's worth noting that the search result included an unusual comment about Gemini, which doesn't seem directly related to the weather. This might be due to the search engine including some astrological information or a joke in its results. However, for the purpose of weather information, we can focus on the fact that it's sunny in San Francisco right now.\n\nIs there anything else you'd like to know about the weather in San Francisco or any other location?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-227a042b-dd97-476e-af32-76a3703af5d8', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


As you can see it now looks up the current weather in Sidi Frej (although our dummy search node still returns results for SF because we don't actually do a search in this example, we just return the same "It's sunny in San Francisco ..." result every time).


<a name="docscloudhow-toshuman_in_the_loop_review_tool_callsmd"></a>

## Review Tool Calls

Human-in-the-loop (HIL) interactions are crucial for [agentic systems](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#human-in-the-loop). A common pattern is to add some human in the loop step after certain tool calls. These tool calls often lead to either a function call or saving of some information. Examples include:

- A tool call to execute SQL, which will then be run by the tool
- A tool call to generate a summary, which will then be saved to the State of the graph

Note that using tool calls is common **whether actually calling tools or not**.

There are typically a few different interactions you may want to do here:

1. Approve the tool call and continue
2. Modify the tool call manually and then continue
3. Give natural language feedback, and then pass that back to the agent instead of continuing

We can implement this in LangGraph using a [breakpoint](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/): breakpoints allow us to interrupt graph execution before a specific step. At this breakpoint, we can manually update the graph state taking one of the three options above

### Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](#simple-usage) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input. 

#### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:


=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Example with no review

Let's look at an example when no review is required (because no tools are called)

=== "Python"

    ```python
    input = { 'messages':[{ "role":"user", "content":"hi!" }] }

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
        interrupt_before=["action"],
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "hi!" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
        interruptBefore: ["action"],
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"hi!\"}]},
       \"stream_mode\": [
         \"updates\"
       ],
       \"interrupt_before\": [\"action\"]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'messages': [{'content': 'hi!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '39c51f14-2d5c-4690-883a-d940854b1845', 'example': False}]}
    {'messages': [{'content': 'hi!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '39c51f14-2d5c-4690-883a-d940854b1845', 'example': False}, {'content': [{'text': "Hello! Welcome. How can I assist you today? Is there anything specific you'd like to know or any information you're looking for?", 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-d65e07fb-43ff-4d98-ab6b-6316191b9c8b', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 355, 'output_tokens': 31, 'total_tokens': 386}}]}


If we check the state, we can see that it is finished

=== "Python"

    ```python
    state = await client.threads.get_state(thread["thread_id"])

    print(state['next'])
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);

    console.log(state.next);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | jq -c '.next'
    ```

Output:

    []

### Example of approving tool

Let's now look at what it looks like to approve a tool call. Note that we don't need to pass an interrupt to our streaming calls because the graph (defined [here](#simple-usage)) was already compiled with an interrupt before the `human_review_node`.

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "what's the weather in sf?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]}
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '54e19d6e-89fa-44fb-b92c-12e7dd4ddf08', 'example': False}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '54e19d6e-89fa-44fb-b92c-12e7dd4ddf08', 'example': False}, {'content': [{'text': "Certainly! I can help you check the weather in San Francisco. To get this information, I'll use the weather search function. Let me do that for you right away.", 'type': 'text', 'index': 0}, {'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-45a6b6c3-ac69-42a4-8957-d982203d6392', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 90, 'total_tokens': 450}}]}


If we now check, we can see that it is waiting on human review:

=== "Python"

    ```python
    state = await client.threads.get_state(thread["thread_id"])

    print(state['next'])
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);

    console.log(state.next);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DELPOYMENT_URL>/threads/<THREAD_ID>/state | jq -c '.next'
    ```

Output:

    ['human_review_node']

To approve the tool call, we can just continue the thread with no edits. To do this, we just create a new run with no inputs.

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
        stream_mode="values",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
        streamMode: "values",
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\"
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '54e19d6e-89fa-44fb-b92c-12e7dd4ddf08', 'example': False}, {'content': [{'text': "Certainly! I can help you check the weather in San Francisco. To get this information, I'll use the weather search function. Let me do that for you right away.", 'type': 'text', 'index': 0}, {'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-45a6b6c3-ac69-42a4-8957-d982203d6392', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 90, 'total_tokens': 450}}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '826cd0f2-9cc6-46f0-b7df-daa6a05d13d2', 'tool_call_id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'artifact': None, 'status': 'success'}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '54e19d6e-89fa-44fb-b92c-12e7dd4ddf08', 'example': False}, {'content': [{'text': "Certainly! I can help you check the weather in San Francisco. To get this information, I'll use the weather search function. Let me do that for you right away.", 'type': 'text', 'index': 0}, {'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-45a6b6c3-ac69-42a4-8957-d982203d6392', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 90, 'total_tokens': 450}}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '826cd0f2-9cc6-46f0-b7df-daa6a05d13d2', 'tool_call_id': 'toolu_015yrR3GMDXe6X8m2p9CsEDN', 'artifact': None, 'status': 'success'}, {'content': [{'text': "\n\nGreat news! The weather in San Francisco is sunny today. It's a beautiful day in the city by the bay. Is there anything else you'd like to know about the weather or any other information I can help you with?", 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-5d5fd0f1-a939-447e-801a-9aaa812322d3', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 464, 'output_tokens': 50, 'total_tokens': 514}}]}

### Edit Tool Call

Let's now say we want to edit the tool call. E.g. change some of the parameters (or even the tool called!) but then execute that tool.

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="values",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "what's the weather in sf?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "values",
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]}
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'cec11391-84da-464b-bd2a-bd4f0d93b9ee', 'example': False}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'cec11391-84da-464b-bd2a-bd4f0d93b9ee', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01SunSpDurNfcnXppWLPrtjC', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-6326da9f-6061-4e12-8586-482e32ab4cab', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01SunSpDurNfcnXppWLPrtjC', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}]}


To do this, we first need to update the state. We can do this by passing a message in with the **same** id of the message we want to overwrite. This will have the effect of **replacing** that old message. Note that this is only possible because of the **reducer** we are using that replaces messages with the same ID - read more about that [here](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state).


=== "Python"

    ```python
    # To get the ID of the message we want to replace, we need to fetch the current state and find it there.
    state = await client.threads.get_state(thread['thread_id'])
    print("Current State:")
    print(state['values'])
    print("\nCurrent Tool Call ID:")
    current_content = state['values']['messages'][-1]['content']
    current_id = state['values']['messages'][-1]['id']
    tool_call_id = state['values']['messages'][-1]['tool_calls'][0]['id']
    print(tool_call_id)

    # We now need to construct a replacement tool call.
    # We will change the argument to be `San Francisco, USA`
    # Note that we could change any number of arguments or tool names - it just has to be a valid one
    new_message = {
        "role": "assistant", 
        "content": current_content,
        "tool_calls": [
            {
                "id": tool_call_id,
                "name": "weather_search",
                "args": {"city": "San Francisco, USA"}
            }
        ],
        # This is important - this needs to be the same as the message you replacing!
        # Otherwise, it will show up as a separate message
        "id": current_id
    }
    await client.threads.update_state(
        # This is the config which represents this thread
        thread['thread_id'], 
        # This is the updated value we want to push
        {"messages": [new_message]}, 
        # We push this update acting as our human_review_node
        as_node="human_review_node"
    )

    print("\nResuming Execution")
    # Let's now continue executing from here
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread.thread_id);
    console.log("Current State:");
    console.log(state.values);

    console.log("\nCurrent Tool Call ID:");
    const lastMessage = state.values.messages[state.values.messages.length - 1];
    const currentContent = lastMessage.content;
    const currentId = lastMessage.id;
    const toolCallId = lastMessage.tool_calls[0].id;
    console.log(toolCallId);

    // Construct a replacement tool call
    const newMessage = {
      role: "assistant",
      content: currentContent,
      tool_calls: [
        {
          id: toolCallId,
          name: "weather_search",
          args: { city: "San Francisco, USA" }
        }
      ],
      // Ensure the ID is the same as the message you're replacing
      id: currentId
    };

    await client.threads.updateState(
      thread.thread_id,  // Thread ID
      {
        values: { "messages": [newMessage] },  // Updated message
        asNode: "human_review_node"
      }  // Acting as human_review_node
    );

    console.log("\nResuming Execution");
    // Continue executing from here
    const streamResponseResumed = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
      }
    );

    for await (const chunk of streamResponseResumed) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
    --header 'Content-Type: application/json' \
    --data "{
        \"values\": { \"messages\": [$(curl --request GET \
            --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state |
            jq -c '{
            role: "assistant",
            content: .values.messages[-1].content,
            tool_calls: [
                {
                id: .values.messages[-1].tool_calls[0].id,
                name: "weather_search",
                args: { city: "San Francisco, USA" }
                }
            ],
            id: .values.messages[-1].id
            }')
        ]},
        \"as_node\": \"human_review_node\"
    }" && echo "Resuming Execution" && curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data '{
    "assistant_id": "agent"
    }' | \
    sed 's/\r$//' | \
    awk '
    /^event:/ {
        if (data_content != "" && event_type != "metadata") {
            print data_content "\n"
        }
        sub(/^event: /, "", $0)
        event_type = $0
        data_content = ""
    }
    /^data:/ {
        sub(/^data: /, "", $0)
        data_content = $0
    }
    END {
        if (data_content != "" && event_type != "metadata") {
            print data_content "\n"
        }
    }
    '
    ```

Output:

    Current State:
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '8713d1fa-9b26-4eab-b768-dafdaac70590', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-ede13f26-daf5-4d8f-817a-7611075bbcf1', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}]}

    Current Tool Call ID:
    toolu_01VzagzsUGZsNMwW1wHkcw7h

    Resuming Execution
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '8713d1fa-9b26-4eab-b768-dafdaac70590', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ede13f26-daf5-4d8f-817a-7611075bbcf1', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '7fc7d463-66bf-4555-9929-6af483de169b', 'tool_call_id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'artifact': None, 'status': 'success'}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '8713d1fa-9b26-4eab-b768-dafdaac70590', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ede13f26-daf5-4d8f-817a-7611075bbcf1', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '7fc7d463-66bf-4555-9929-6af483de169b', 'tool_call_id': 'toolu_01VzagzsUGZsNMwW1wHkcw7h', 'artifact': None, 'status': 'success'}, {'content': [{'text': "\n\nBased on the search result, the weather in San Francisco is sunny! It's a beautiful day in the city by the bay. Is there anything else you'd like to know about the weather or any other information I can help you with?", 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-d90ce97a-39f9-4330-985e-67c5f351a0c5', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 455, 'output_tokens': 52, 'total_tokens': 507}}]}

### Give feedback to a tool call

Sometimes, you may not want to execute a tool call, but you also may not want to ask the user to manually modify the tool call. In that case it may be better to get natural language feedback from the user. You can then insert these feedback as a mock **RESULT** of the tool call.

There are multiple ways to do this:

You could add a new message to the state (representing the "result" of a tool call)
You could add TWO new messages to the state - one representing an "error" from the tool call, other HumanMessage representing the feedback
Both are similar in that they involve adding messages to the state. The main difference lies in the logic AFTER the `human_node` and how it handles different types of messages.

For this example we will just add a single tool call representing the feedback. Let's see this in action!

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf?"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "what's the weather in sf?" }] };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in sf?\"}]}
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c80f13d0-674d-4233-b6a0-3940509d3cf3', 'example': False}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c80f13d0-674d-4233-b6a0-3940509d3cf3', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_016XyTdFA8NuPWeLyZPSzoM3', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-4911ac27-3d7c-4edf-a3ca-c2908e3922eb', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_016XyTdFA8NuPWeLyZPSzoM3', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}]}

To do this, we first need to update the state. We can do this by passing a message in with the same **tool call id** of the tool call we want to respond to. Note that this is a **different*** ID from above

=== "Python"

    ```python
    # To get the ID of the message we want to replace, we need to fetch the current state and find it there.
    state = await client.threads.get_state(thread['thread_id'])
    print("Current State:")
    print(state['values'])
    print("\nCurrent Tool Call ID:")
    tool_call_id = state['values']['messages'][-1]['tool_calls'][0]['id']
    print(tool_call_id)

    # We now need to construct a replacement tool call.
    # We will change the argument to be `San Francisco, USA`
    # Note that we could change any number of arguments or tool names - it just has to be a valid one
    new_message = {
        "role": "tool", 
        # This is our natural language feedback
        "content": "User requested changes: pass in the country as well",
        "name": "weather_search",
        "tool_call_id": tool_call_id
    }
    await client.threads.update_state(
        # This is the config which represents this thread
        thread['thread_id'], 
        # This is the updated value we want to push
        {"messages": [new_message]}, 
        # We push this update acting as our human_review_node
        as_node="human_review_node"
    )

    print("\nResuming execution")
    # Let's now continue executing from here
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
        stream_mode="values",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread.thread_id);
    console.log("Current State:");
    console.log(state.values);

    console.log("\nCurrent Tool Call ID:");
    const lastMessage = state.values.messages[state.values.messages.length - 1];
    const toolCallId = lastMessage.tool_calls[0].id;
    console.log(toolCallId);

    // Construct a replacement tool call
    const newMessage = {
      role: "tool",
      content: "User requested changes: pass in the country as well",
      name: "weather_search",
      tool_call_id: toolCallId,
    };

    await client.threads.updateState(
      thread.thread_id,  // Thread ID
      {
        values: { "messages": [newMessage] },  // Updated message
        asNode: "human_review_node"
      }  // Acting as human_review_node
    );

    console.log("\nResuming Execution");
    // Continue executing from here
    const streamResponseEdited = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
        streamMode: "values",
        interruptBefore: ["action"],
      }
    );

    for await (const chunk of streamResponseEdited) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
    --header 'Content-Type: application/json' \
    --data "{
        \"values\": { \"messages\": [$(curl --request GET \
            --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state |
            jq -c '{
            role: "tool",
            content: "User requested changes: pass in the country as well",
            name: "get_weather",
            tool_call_id: .values.messages[-1].id.tool_calls[0].id
            }')
        ]},
        \"as_node\": \"human_review_node\"
    }" && echo "Resuming Execution" && curl --request POST \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
    --header 'Content-Type: application/json' \
    --data '{
    "assistant_id": "agent"
    }' | \
    sed 's/\r$//' | \
    awk '
    /^event:/ {
        if (data_content != "" && event_type != "metadata") {
            print data_content "\n"
        }
        sub(/^event: /, "", $0)
        event_type = $0
        data_content = ""
    }
    /^data:/ {
        sub(/^data: /, "", $0)
        data_content = $0
    }
    END {
        if (data_content != "" && event_type != "metadata") {
            print data_content "\n"
        }
    }
    '
    ```


Output:

    Current State:
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '3b2bbc38-d11b-49eb-80c0-c24a40dab5a8', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-c5a50900-abf5-4885-9cdb-da2bf0d892ac', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}]}

    Current Tool Call ID:
    toolu_01NNw18j57GEGPZvsa9f1wvX

    Resuming execution
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '3b2bbc38-d11b-49eb-80c0-c24a40dab5a8', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-c5a50900-abf5-4885-9cdb-da2bf0d892ac', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}, {'content': 'User requested changes: pass in the country as well', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '787288be-213c-4fd3-8503-4a009bdb1b00', 'tool_call_id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'artifact': None, 'status': 'success'}, {'content': [{'text': '\n\nI apologize for the oversight. It seems the function requires additional information. Let me try again with a more specific request.', 'type': 'text', 'index': 0}, {'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco, USA"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-5c355a56-cfe3-4046-b49f-f5b09fc397ef', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 461, 'output_tokens': 83, 'total_tokens': 544}}]}

We can see that we now get to another breakpoint - because it went back to the model and got an entirely new prediction of what to call. Let's now approve this one and continue

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const streamResponseResumed = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
      }
    );

    for await (const chunk of streamResponseResumed) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\"
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '3b2bbc38-d11b-49eb-80c0-c24a40dab5a8', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-c5a50900-abf5-4885-9cdb-da2bf0d892ac', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}, {'content': 'User requested changes: pass in the country as well', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '787288be-213c-4fd3-8503-4a009bdb1b00', 'tool_call_id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'artifact': None, 'status': 'success'}, {'content': [{'text': '\n\nI apologize for the oversight. It seems the function requires additional information. Let me try again with a more specific request.', 'type': 'text', 'index': 0}, {'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco, USA"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-5c355a56-cfe3-4046-b49f-f5b09fc397ef', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 461, 'output_tokens': 83, 'total_tokens': 544}}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '3b857482-bca2-4a73-a9ab-1f35a3e43e5f', 'tool_call_id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'artifact': None, 'status': 'success'}]}
    {'messages': [{'content': "what's the weather in sf?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '3b2bbc38-d11b-49eb-80c0-c24a40dab5a8', 'example': False}, {'content': [{'text': 'To get the weather information for San Francisco, I can use the weather_search function. Let me do that for you.', 'type': 'text', 'index': 0}, {'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-c5a50900-abf5-4885-9cdb-da2bf0d892ac', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco'}, 'id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 360, 'output_tokens': 80, 'total_tokens': 440}}, {'content': 'User requested changes: pass in the country as well', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '787288be-213c-4fd3-8503-4a009bdb1b00', 'tool_call_id': 'toolu_01NNw18j57GEGPZvsa9f1wvX', 'artifact': None, 'status': 'success'}, {'content': [{'text': '\n\nI apologize for the oversight. It seems the function requires additional information. Let me try again with a more specific request.', 'type': 'text', 'index': 0}, {'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'input': {}, 'name': 'weather_search', 'type': 'tool_use', 'index': 1, 'partial_json': '{"city": "San Francisco, USA"}'}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'tool_use', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-5c355a56-cfe3-4046-b49f-f5b09fc397ef', 'example': False, 'tool_calls': [{'name': 'weather_search', 'args': {'city': 'San Francisco, USA'}, 'id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'type': 'tool_call'}], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 461, 'output_tokens': 83, 'total_tokens': 544}}, {'content': 'Sunny!', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'weather_search', 'id': '3b857482-bca2-4a73-a9ab-1f35a3e43e5f', 'tool_call_id': 'toolu_01YAbLBoKozJyRQnB8LUMpXC', 'artifact': None, 'status': 'success'}, {'content': [{'text': "\n\nGreat news! The weather in San Francisco is sunny today. Is there anything else you'd like to know about the weather or any other information I can help you with?", 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-6a857bb1-f65b-4b86-93d6-c025e003c777', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 557, 'output_tokens': 38, 'total_tokens': 595}}]}


<a name="docscloudhow-toshuman_in_the_loop_time_travelmd"></a>

## How to Replay and Branch from Prior States

With LangGraph Cloud you have the ability to return to any of your prior states and either re-run the graph to reproduce issues noticed during testing, or branch out in a different way from what was originally done in the prior states. In this guide we will show a quick example of how to rerun past states and how to branch off from previous states as well.

### Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](#build-the-agent) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input. 

#### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Replay a state

#### Initial invocation

Before replaying a state - we need to create states to replay from! In order to do this, let's invoke our graph with a simple message:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "Please search the weather in SF"}]}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const input = { "messages": [{ "role": "user", "content": "Please search the weather in SF" }] }

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"Please search the weather in SF\"}]},
       \"stream_mode\": [
         \"updates\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```
    
Output:

    {'agent': {'messages': [{'content': [{'text': "Certainly! I'll use the search function to look up the current weather in San Francisco for you. Let me do that now.", 'type': 'text'}, {'id': 'toolu_011vroKUtWU7SBdrngpgpFMn', 'input': {'query': 'current weather in San Francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ee639877-d97d-40f8-96dc-d0d1ae22d203', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in San Francisco'}, 'id': 'toolu_011vroKUtWU7SBdrngpgpFMn'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}
    {'action': {'messages': [{'content': '["I looked up: current weather in San Francisco. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini ­ƒÿê."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '7bad0e72-5ebe-4b08-9b8a-b99b0fe22fb7', 'tool_call_id': 'toolu_011vroKUtWU7SBdrngpgpFMn'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, I can provide you with information about the current weather in San Francisco:\n\nThe weather in San Francisco is currently sunny. This is great news for outdoor activities and enjoying the city's beautiful sights.\n\nIt's worth noting that the search result included an unusual comment about Geminis, which isn't typically part of a weather report. This might be due to the search engine including some astrological information or a joke in its results. However, for the purpose of answering your question about the weather, we can focus on the fact that it's sunny in San Francisco.\n\nIf you need any more specific information about the weather in San Francisco, such as temperature, wind speed, or forecast for the coming days, please let me know, and I'd be happy to search for that information for you.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-dbac539a-33c8-4f0c-9e20-91f318371e7c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


Now let's get our list of states, and invoke from the third state (right before the tool get called):


=== "Python"

    ```python
    states = await client.threads.get_history(thread['thread_id'])

    # We can confirm that this state is correct by checking the 'next' attribute and seeing that it is the tool call node
    state_to_replay = states[2]
    print(state_to_replay['next'])
    ```

=== "Javascript"

    ```js
    const states = await client.threads.getHistory(thread['thread_id']);

    // We can confirm that this state is correct by checking the 'next' attribute and seeing that it is the tool call node
    const stateToReplay = states[2];
    console.log(stateToReplay['next']);
    ```

=== "CURL"

    ```bash
    curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/history | jq -r '.[2].next'
    ```

Output:

    ['action']



To rerun from a state, we need first issue an empty update to the thread state. Then we need to pass in the resulting `checkpoint_id` as follows:

=== "Python"

    ```python
    state_to_replay = states[2]
    updated_config = await client.threads.update_state(
        thread["thread_id"],
        {"messages": []},
        checkpoint_id=state_to_replay["checkpoint_id"]
    )
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id, # graph_id
        input=None,
        stream_mode="updates",
        checkpoint_id=updated_config["checkpoint_id"]
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const stateToReplay = states[2];
    const config = await client.threads.updateState(thread["thread_id"], { values: {"messages": [] }, checkpointId: stateToReplay["checkpoint_id"] });
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
        streamMode: "updates",
        checkpointId: config["checkpoint_id"]
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/history | jq -c '
        .[2] as $state_to_replay |
        {
            values: { messages: .[2].values.messages[-1] },
            checkpoint_id: $state_to_replay.checkpoint_id
        }' | \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
        --header 'Content-Type: application/json' \
        --data @- | jq .checkpoint_id | \
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"checkpoint_id\": \"$1\",
       \"stream_mode\": [
         \"updates\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'action': {'messages': [{'content': '["I looked up: current weather in San Francisco. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini ­ƒÿê."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': 'eba650e5-400e-4938-8508-f878dcbcc532', 'tool_call_id': 'toolu_011vroKUtWU7SBdrngpgpFMn'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, I can provide you with information about the current weather in San Francisco:\n\nThe weather in San Francisco is currently sunny. This is great news if you're planning any outdoor activities or simply want to enjoy a pleasant day in the city.\n\nIt's worth noting that the search result included an unusual comment about Geminis, which doesn't seem directly related to the weather. This appears to be a playful or humorous addition to the weather report, possibly from the source where this information was obtained.\n\nIs there anything else you'd like to know about the weather in San Francisco or any other information you need?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-bc6dca3f-a1e2-4f59-a69b-fe0515a348bb', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


As we can see, the graph restarted from the tool node with the same input as our original graph run.

### Branch off from previous state

Using LangGraph's checkpointing, you can do more than just replay past states. You can branch off previous locations to let the agent explore alternate trajectories or to let a user "version control" changes in a workflow.

Let's show how to do this to edit the state at a particular point in time. Let's update the state to change the input to the tool

=== "Python"

    ```python
    # Let's now get the last message in the state
    # This is the one with the tool calls that we want to update
    last_message = state_to_replay['values']['messages'][-1]

    # Let's now update the args for that tool call
    last_message['tool_calls'][0]['args'] = {'query': 'current weather in SF'}

    config = await client.threads.update_state(thread['thread_id'],{"messages":[last_message]},checkpoint_id=state_to_replay['checkpoint_id'])
    ```

=== "Javascript"

    ```js
    // Let's now get the last message in the state
    // This is the one with the tool calls that we want to update
    let lastMessage = stateToReplay['values']['messages'][-1];

    // Let's now update the args for that tool call
    lastMessage['tool_calls'][0]['args'] = { 'query': 'current weather in SF' };

    const config = await client.threads.updateState(thread['thread_id'], { values: { "messages": [lastMessage] }, checkpointId: stateToReplay['checkpoint_id'] });
    ```

=== "CURL"

    ```bash
    curl -s --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/history | \
    jq -c '
        .[2] as $state_to_replay |
        .[2].values.messages[-1].tool_calls[0].args.query = "current weather in SF" |
        {
            values: { messages: .[2].values.messages[-1] },
            checkpoint_id: $state_to_replay.checkpoint_id
        }' | \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
        --header 'Content-Type: application/json' \
        --data @-
    ```

Now we can rerun our graph with this new config, starting from the `new_state`, which is a branch of our `state_to_replay`:

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
        stream_mode="updates",
        checkpoint_id=config['checkpoint_id']
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
        streamMode: "updates",
        checkpointId: config['checkpoint_id'],
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl -s --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | \
    jq -c '.checkpoint_id' | \
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"checkpoint_id\": \"$1\",
       \"stream_mode\": [
         \"updates\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:


    {'action': {'messages': [{'content': '["I looked up: current weather in SF. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini ­ƒÿê."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '2baf9941-4fda-4081-9f87-d76795d289f1', 'tool_call_id': 'toolu_011vroKUtWU7SBdrngpgpFMn'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, I can provide you with information about the current weather in San Francisco (SF):\n\nThe weather in San Francisco is currently sunny. This means it's a clear day with plenty of sunshine. \n\nIt's worth noting that the specific temperature wasn't provided in the search result, but sunny weather in San Francisco typically means comfortable temperatures. San Francisco is known for its mild climate, so even on sunny days, it's often not too hot.\n\nThe search result also included a playful reference to astrological signs, mentioning Gemini. However, this is likely just a joke or part of the search engine's presentation and not related to the actual weather conditions.\n\nIs there any specific information about the weather in San Francisco you'd like to know more about? I'd be happy to perform another search if you need details on temperature, wind conditions, or the forecast for the coming days.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-a83de52d-ed18-4402-9384-75c462485743', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


As we can see, the search query changed from San Francisco to SF, just as we had hoped!


<a name="docscloudhow-toshuman_in_the_loop_user_inputmd"></a>

## How to Wait for User Input

One of the main human-in-the-loop interaction patterns is waiting for human input. A key use case involves asking the user clarifying questions. One way to accomplish this is simply go to the `END` node and exit the graph. Then, any user response comes back in as fresh invocation of the graph. This is basically just creating a chatbot architecture.

The issue with this is it is tough to resume back in a particular point in the graph. Often times the agent is halfway through some process, and just needs a bit of a user input. Although it is possible to design your graph in such a way where you have a `conditional_entry_point` to route user messages back to the right place, that is not super scalable (as it essentially involves having a routing function that can end up almost anywhere).

A separate way to do this is to have a node explicitly for getting user input. This is easy to implement in a notebook setting - you just put an `input()` call in the node. But that isn't exactly production ready.

Luckily, LangGraph makes it possible to do similar things in a production way. The basic idea is:

- Set up a node that represents human input. This can have specific incoming/outgoing edges (as you desire). There shouldn't actually be any logic inside this node.
- Add a breakpoint before the node. This will stop the graph before this node executes (which is good, because there's no real logic in it anyways)
- Use `.update_state` to update the state of the graph. Pass in whatever human response you get. The key here is to use the `as_node` parameter to apply this update **as if you were that node**. This will have the effect of making it so that when you resume execution next it resumes as if that node just acted, and not from the beginning.

### Setup

We are not going to show the full code for the graph we are hosting, but you can see it [here](#agent) if you want to. Once this graph is hosted, we are ready to invoke it and wait for user input.

#### SDK initialization

First, we need to setup our client so that we can communicate with our hosted graph:

=== "Python"

    ```python
    from langgraph_sdk import get_client
    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Waiting for user input

#### Initial invocation

Now, let's invoke our graph by interrupting before `ask_human` node:

=== "Python"

    ```python
    input = {
        "messages": [
            {
                "role": "user",
                "content": "Use the search tool to ask the user where they are, then look up the weather there",
            }
        ]
    }

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
        interrupt_before=["ask_human"],
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```
=== "Javascript"

    ```js
    const input = {
      messages: [
        {
          role: "human",
          content: "Use the search tool to ask the user where they are, then look up the weather there"
        }
      ]
    };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: input,
        streamMode: "updates",
        interruptBefore: ["ask_human"]
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"Use the search tool to ask the user where they are, then look up the weather there\"}]},
       \"interrupt_before\": [\"ask_human\"],
       \"stream_mode\": [
         \"updates\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```
    
Output:

    {'agent': {'messages': [{'content': [{'text': "Certainly! I'll use the AskHuman function to ask the user about their location, and then I'll use the search function to look up the weather for that location. Let's start by asking the user where they are.", 'type': 'text'}, {'id': 'toolu_01RFahzYPvnPWTb2USk2RdKR', 'input': {'question': 'Where are you currently located?'}, 'name': 'AskHuman', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-a8422215-71d3-4093-afb4-9db141c94ddb', 'example': False, 'tool_calls': [{'name': 'AskHuman', 'args': {'question': 'Where are you currently located?'}, 'id': 'toolu_01RFahzYPvnPWTb2USk2RdKR'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}


#### Adding user input to state

We now want to update this thread with a response from the user. We then can kick off another run.

Because we are treating this as a tool call, we will need to update the state as if it is a response from a tool call. In order to do this, we will need to check the state to get the ID of the tool call.


=== "Python"

    ```python
    state = await client.threads.get_state(thread['thread_id'])
    tool_call_id = state['values']['messages'][-1]['tool_calls'][0]['id']

    # We now create the tool call with the id and the response we want
    tool_message = [{"tool_call_id": tool_call_id, "type": "tool", "content": "san francisco"}]

    await client.threads.update_state(thread['thread_id'], {"messages": tool_message}, as_node="ask_human")
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);
    const toolCallId = state.values.messages[state.values.messages.length - 1].tool_calls[0].id;

    // We now create the tool call with the id and the response we want
    const toolMessage = [
      {
        tool_call_id: toolCallId,
        type: "tool",
        content: "san francisco"
      }
    ];

    await client.threads.updateState(
      thread["thread_id"],
      { values: { messages: toolMessage } },
      { asNode: "ask_human" }
    );
    ```

=== "CURL"

    ```bash
    curl --request GET \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
     | jq -r '.values.messages[-1].tool_calls[0].id' \
     | sh -c '
         TOOL_CALL_ID="$1"
         
         # Construct the JSON payload
         JSON_PAYLOAD=$(printf "{\"messages\": [{\"tool_call_id\": \"%s\", \"type\": \"tool\", \"content\": \"san francisco\"}], \"as_node\": \"ask_human\"}" "$TOOL_CALL_ID")
         
         # Send the updated state
         curl --request POST \
              --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state \
              --header "Content-Type: application/json" \
              --data "${JSON_PAYLOAD}"
     ' _ 
    ```

Output:

    {'configurable': {'thread_id': 'a9f322ae-4ed1-41ec-942b-38cb3d342c3a',
    'checkpoint_ns': '',
    'checkpoint_id': '1ef58e97-a623-63dd-8002-39a9a9b20be3'}}


#### Invoking after receiving human input

We can now tell the agent to continue. We can just pass in None as the input to the graph, since no additional input is needed:

=== "Python"

    ```python
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=None,
        stream_mode="updates",
    ):
        if chunk.data and chunk.event != "metadata": 
            print(chunk.data)
    ```
=== "Javascript"

    ```js
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantId,
      {
        input: null,
        streamMode: "updates"
      }
    );

    for await (const chunk of streamResponse) {
      if (chunk.data && chunk.event !== "metadata") {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \                                                                             
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"stream_mode\": [
         \"updates\"
       ]
     }"| \ 
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
         sub(/^event: /, "", $0)
         event_type = $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "" && event_type != "metadata") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    {'agent': {'messages': [{'content': [{'text': "Thank you for letting me know that you're in San Francisco. Now, I'll use the search function to look up the weather in San Francisco.", 'type': 'text'}, {'id': 'toolu_01K57ofmgG2wyJ8tYJjbq5k7', 'input': {'query': 'current weather in San Francisco'}, 'name': 'search', 'type': 'tool_use'}], 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-241baed7-db5e-44ce-ac3c-56431705c22b', 'example': False, 'tool_calls': [{'name': 'search', 'args': {'query': 'current weather in San Francisco'}, 'id': 'toolu_01K57ofmgG2wyJ8tYJjbq5k7'}], 'invalid_tool_calls': [], 'usage_metadata': None}]}}
    {'action': {'messages': [{'content': '["I looked up: current weather in San Francisco. Result: It\'s sunny in San Francisco, but you better look out if you\'re a Gemini ­ƒÿê."]', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': 'search', 'id': '8b699b95-8546-4557-8e66-14ea71a15ed8', 'tool_call_id': 'toolu_01K57ofmgG2wyJ8tYJjbq5k7'}]}}
    {'agent': {'messages': [{'content': "Based on the search results, I can provide you with information about the current weather in San Francisco:\n\nThe weather in San Francisco is currently sunny. It's a beautiful day in the city! \n\nHowever, I should note that the search result included an unusual comment about Gemini zodiac signs. This appears to be either a joke or potentially irrelevant information added by the search engine. For accurate and detailed weather information, you might want to check a reliable weather service or app for San Francisco.\n\nIs there anything else you'd like to know about the weather or San Francisco?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-b4d7309f-f849-46aa-b6ef-475bcabd2be9', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}



<a name="docscloudhow-tosinterrupt_concurrentmd"></a>

## Interrupt

This guide assumes knowledge of what double-texting is, which you can learn about in the [double-texting conceptual guide](#docsconceptsdouble_textingmd).

The guide covers the `interrupt` option for double texting, which interrupts the prior run of the graph and starts a new one with the double-text. This option does not delete the first run, but rather keeps it in the database but sets its status to `interrupted`. Below is a quick example of using the `interrupt` option.

### Setup

First, we will define a quick helper function for printing out JS and CURL model outputs (you can skip this if using Python):

=== "Javascript"

    ```js
    function prettyPrint(m) {
      const padded = " " + m['type'] + " ";
      const sepLen = Math.floor((80 - padded.length) / 2);
      const sep = "=".repeat(sepLen);
      const secondSep = sep + (padded.length % 2 ? "=" : "");
      
      console.log(`${sep}${padded}${secondSep}`);
      console.log("\n\n");
      console.log(m.content);
    }
    ```

=== "CURL"

    ```bash
    # PLACE THIS IN A FILE CALLED pretty_print.sh
    pretty_print() {
      local type="$1"
      local content="$2"
      local padded=" $type "
      local total_width=80
      local sep_len=$(( (total_width - ${#padded}) / 2 ))
      local sep=$(printf '=%.0s' $(eval "echo {1.."${sep_len}"}"))
      local second_sep=$sep
      if (( (total_width - ${#padded}) % 2 )); then
        second_sep="${second_sep}="
      fi

      echo "${sep}${padded}${second_sep}"
      echo
      echo "$content"
    }
    ```

Now, let's import our required packages and instantiate our client, assistant, and thread.

=== "Python"

    ```python
    import asyncio

    from langchain_core.messages import convert_to_messages
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Create runs

Now we can start our two runs and join the second one until it has completed:

=== "Python"

    ```python
    # the first run will be interrupted
    interrupted_run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    )
    # sleep a bit to get partial outputs from the first run
    await asyncio.sleep(2)
    run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": "what's the weather in nyc?"}]},
        multitask_strategy="interrupt",
    )
    # wait until the second run completes
    await client.runs.join(thread["thread_id"], run["run_id"])
    ```

=== "Javascript"

    ```js
    // the first run will be interrupted
    let interruptedRun = await client.runs.create(
      thread["thread_id"],
      assistantId,
      { input: { messages: [{ role: "human", content: "what's the weather in sf?" }] } }
    );
    // sleep a bit to get partial outputs from the first run
    await new Promise(resolve => setTimeout(resolve, 2000)); 

    let run = await client.runs.create(
      thread["thread_id"],
      assistantId,
      { 
        input: { messages: [{ role: "human", content: "what's the weather in nyc?" }] },
        multitaskStrategy: "interrupt" 
      }
    );

    // wait until the second run completes
    await client.runs.join(thread["thread_id"], run["run_id"]);
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in sf?\"}]},
    }" && sleep 2 && curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in nyc?\"}]},
      \"multitask_strategy\": \"interrupt\"
    }" && curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/join
    ```

### View run results

We can see that the thread has partial data from the first run + data from the second run


=== "Python"

    ```python
    state = await client.threads.get_state(thread["thread_id"])

    for m in convert_to_messages(state["values"]["messages"]):
        m.pretty_print()
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);

    for (const m of state['values']['messages']) {
      prettyPrint(m);
    }
    ```

=== "CURL"

    ```bash
    source pretty_print.sh && curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | \
    jq -c '.values.messages[]' | while read -r element; do
        type=$(echo "$element" | jq -r '.type')
        content=$(echo "$element" | jq -r '.content | if type == "array" then tostring else . end')
        pretty_print "$type" "$content"
    done
    ```

Output:

    ================================ Human Message =================================
    
    what's the weather in sf?
    ================================== Ai Message ==================================
    
    [{'id': 'toolu_01MjNtVJwEcpujRGrf3x6Pih', 'input': {'query': 'weather in san francisco'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01MjNtVJwEcpujRGrf3x6Pih)
     Call ID: toolu_01MjNtVJwEcpujRGrf3x6Pih
      Args:
        query: weather in san francisco
    ================================= Tool Message =================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.wunderground.com/hourly/us/ca/san-francisco/KCASANFR2002/date/2024-6-18", "content": "High 64F. Winds W at 10 to 20 mph. A few clouds from time to time. Low 49F. Winds W at 10 to 20 mph. Temp. San Francisco Weather Forecasts. Weather Underground provides local & long-range weather ..."}]
    ================================ Human Message =================================
    
    what's the weather in nyc?
    ================================== Ai Message ==================================
    
    [{'id': 'toolu_01KtE1m1ifPLQAx4fQLyZL9Q', 'input': {'query': 'weather in new york city'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01KtE1m1ifPLQAx4fQLyZL9Q)
     Call ID: toolu_01KtE1m1ifPLQAx4fQLyZL9Q
      Args:
        query: weather in new york city
    ================================= Tool Message =================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.accuweather.com/en/us/new-york/10021/june-weather/349727", "content": "Get the monthly weather forecast for New York, NY, including daily high/low, historical averages, to help you plan ahead."}]
    ================================== Ai Message ==================================
    
    The search results provide weather forecasts and information for New York City. Based on the top result from AccuWeather, here are some key details about the weather in NYC:
    
    - This is a monthly weather forecast for New York City for the month of June.
    - It includes daily high and low temperatures to help plan ahead.
    - Historical averages for June in NYC are also provided as a reference point.
    - More detailed daily or hourly forecasts with precipitation chances, humidity, wind, etc. can be found by visiting the AccuWeather page.
    
    So in summary, the search provides a convenient overview of the expected weather conditions in New York City over the next month to give you an idea of what to prepare for if traveling or making plans there. Let me know if you need any other details!


Verify that the original, interrupted run was interrupted

=== "Python"

    ```python
    print((await client.runs.get(thread["thread_id"], interrupted_run["run_id"]))["status"])
    ```

=== "Javascript"

    ```js
    console.log((await client.runs.get(thread['thread_id'], interruptedRun["run_id"]))["status"])
    ```

Output:

    'interrupted'



<a name="docscloudhow-tosinvoke_studiomd"></a>

## Invoke Assistant

The LangGraph Studio lets you test different configurations and inputs to your graph. It also provides a nice visualization of your graph during execution so it is easy to see which nodes are being run and what the outputs of each individual node are.

1. The LangGraph Studio UI displays a visualization of the selected assistant.
    1. In the top-left dropdown menu of the left-hand pane, select an assistant.
    1. In the bottom of the left-hand pane, edit the `Input` and `Configure` the assistant.
    1. Select `Submit` to invoke the selected assistant.
1. View output of the invocation in the right-hand pane.

The following video shows these exact steps being carried out:

<video controls allowfullscreen="true" poster="../img/studio_input_poster.png">
    <source src="../img/studio_input.mp4" type="video/mp4">
</video>


<a name="docscloudhow-tosreject_concurrentmd"></a>

## Reject

This guide assumes knowledge of what double-texting is, which you can learn about in the [double-texting conceptual guide](#docsconceptsdouble_textingmd).

The guide covers the `reject` option for double texting, which rejects the new run of the graph by throwing an error and continues with the original run until completion. Below is a quick example of using the `reject` option.

### Setup

First, we will define a quick helper function for printing out JS and CURL model outputs (you can skip this if using Python):

=== "Javascript"

    ```js
    function prettyPrint(m) {
      const padded = " " + m['type'] + " ";
      const sepLen = Math.floor((80 - padded.length) / 2);
      const sep = "=".repeat(sepLen);
      const secondSep = sep + (padded.length % 2 ? "=" : "");
      
      console.log(`${sep}${padded}${secondSep}`);
      console.log("\n\n");
      console.log(m.content);
    }
    ```

=== "CURL"

    ```bash
    # PLACE THIS IN A FILE CALLED pretty_print.sh
    pretty_print() {
      local type="$1"
      local content="$2"
      local padded=" $type "
      local total_width=80
      local sep_len=$(( (total_width - ${#padded}) / 2 ))
      local sep=$(printf '=%.0s' $(eval "echo {1.."${sep_len}"}"))
      local second_sep=$sep
      if (( (total_width - ${#padded}) % 2 )); then
        second_sep="${second_sep}="
      fi

      echo "${sep}${padded}${second_sep}"
      echo
      echo "$content"
    }
    ```

Now, let's import our required packages and instantiate our client, assistant, and thread.

=== "Python"

    ```python
    import httpx
    from langchain_core.messages import convert_to_messages
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Create runs

Now we can run a thread and try to run a second one with the "reject" option, which should fail since we have already started a run:


=== "Python"

    ```python
    run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    )
    try:
        await client.runs.create(
            thread["thread_id"],
            assistant_id,
            input={
                "messages": [{"role": "user", "content": "what's the weather in nyc?"}]
            },
            multitask_strategy="reject",
        )
    except httpx.HTTPStatusError as e:
        print("Failed to start concurrent run", e)
    ```

=== "Javascript"

    ```js
    const run = await client.runs.create(
      thread["thread_id"],
      assistantId,
      input={"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    );
    
    try {
      await client.runs.create(
        thread["thread_id"],
        assistantId,
        { 
          input: {"messages": [{"role": "user", "content": "what's the weather in nyc?"}]},
          multitask_strategy:"reject"
        },
      );
    } catch (e) {
      console.error("Failed to start concurrent run", e);
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in sf?\"}]},
    }" && curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in nyc?\"}]},
      \"multitask_strategy\": \"reject\"
    }" || { echo "Failed to start concurrent run"; echo "Error: $?" >&2; }
    ```

Output:

    Failed to start concurrent run Client error '409 Conflict' for url 'http://localhost:8123/threads/f9e7088b-8028-4e5c-88d2-9cc9a2870e50/runs'
    For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/409

### View run results

We can verify that the original thread finished executing:

=== "Python"

    ```python
    # wait until the original run completes
    await client.runs.join(thread["thread_id"], run["run_id"])

    state = await client.threads.get_state(thread["thread_id"])

    for m in convert_to_messages(state["values"]["messages"]):
        m.pretty_print()
    ```

=== "Javascript"

    ```js
    await client.runs.join(thread["thread_id"], run["run_id"]);

    const state = await client.threads.getState(thread["thread_id"]);

    for (const m of state["values"]["messages"]) {
      prettyPrint(m);
    }
    ```

=== "CURL"

    ```bash
    source pretty_print.sh && curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/join && \
    curl --request GET --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | \
    jq -c '.values.messages[]' | while read -r element; do
        type=$(echo "$element" | jq -r '.type')
        content=$(echo "$element" | jq -r '.content | if type == "array" then tostring else . end')
        pretty_print "$type" "$content"
    done
    ```

Output:

    ================================ Human Message =================================
    
    what's the weather in sf?
    ================================== Ai Message ==================================
    
    [{'id': 'toolu_01CyewEifV2Kmi7EFKHbMDr1', 'input': {'query': 'weather in san francisco'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01CyewEifV2Kmi7EFKHbMDr1)
     Call ID: toolu_01CyewEifV2Kmi7EFKHbMDr1
      Args:
        query: weather in san francisco
    ================================= Tool Message =================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.accuweather.com/en/us/san-francisco/94103/june-weather/347629", "content": "Get the monthly weather forecast for San Francisco, CA, including daily high/low, historical averages, to help you plan ahead."}]
    ================================== Ai Message ==================================
    
    According to the search results from Tavily, the current weather in San Francisco is:
    
    The average high temperature in San Francisco in June is around 65┬░F (18┬░C), with average lows around 54┬░F (12┬░C). June tends to be one of the cooler and foggier months in San Francisco due to the marine layer of fog that often blankets the city during the summer months.
    
    Some key points about the typical June weather in San Francisco:
    
    - Mild temperatures with highs in the 60s F and lows in the 50s F
    - Foggy mornings that often burn off to sunny afternoons
    - Little to no rainfall, as June falls in the dry season
    - Breezy conditions, with winds off the Pacific Ocean
    - Layers are recommended for changing weather conditions
    
    So in summary, you can expect mild, foggy mornings giving way to sunny but cool afternoons in San Francisco this time of year. The marine layer keeps temperatures moderate compared to other parts of California in June.



<a name="docscloudhow-tosrollback_concurrentmd"></a>

## Rollback

This guide assumes knowledge of what double-texting is, which you can learn about in the [double-texting conceptual guide](#docsconceptsdouble_textingmd).

The guide covers the `rollback` option for double texting, which interrupts the prior run of the graph and starts a new one with the double-text. This option is very similar to the `interrupt` option, but in this case the first run is completely deleted from the database and cannot be restarted. Below is a quick example of using the `rollback` option.

### Setup

First, we will define a quick helper function for printing out JS and CURL model outputs (you can skip this if using Python):

=== "Javascript"

    ```js
    function prettyPrint(m) {
      const padded = " " + m['type'] + " ";
      const sepLen = Math.floor((80 - padded.length) / 2);
      const sep = "=".repeat(sepLen);
      const secondSep = sep + (padded.length % 2 ? "=" : "");
      
      console.log(`${sep}${padded}${secondSep}`);
      console.log("\n\n");
      console.log(m.content);
    }
    ```

=== "CURL"

    ```bash
    # PLACE THIS IN A FILE CALLED pretty_print.sh
    pretty_print() {
      local type="$1"
      local content="$2"
      local padded=" $type "
      local total_width=80
      local sep_len=$(( (total_width - ${#padded}) / 2 ))
      local sep=$(printf '=%.0s' $(eval "echo {1.."${sep_len}"}"))
      local second_sep=$sep
      if (( (total_width - ${#padded}) % 2 )); then
        second_sep="${second_sep}="
      fi

      echo "${sep}${padded}${second_sep}"
      echo
      echo "$content"
    }
    ```

Now, let's import our required packages and instantiate our client, assistant, and thread.

=== "Python"

    ```python
    import asyncio

    import httpx
    from langchain_core.messages import convert_to_messages
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

### Create runs

Now let's run a thread with the multitask parameter set to "rollback":

=== "Python"

    ```python
    # the first run will be rolled back
    rolled_back_run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": "what's the weather in sf?"}]},
    )
    run = await client.runs.create(
        thread["thread_id"],
        assistant_id,
        input={"messages": [{"role": "user", "content": "what's the weather in nyc?"}]},
        multitask_strategy="rollback",
    )
    # wait until the second run completes
    await client.runs.join(thread["thread_id"], run["run_id"])
    ```

=== "Javascript"

    ```js
    // the first run will be interrupted
    let rolledBackRun = await client.runs.create(
      thread["thread_id"],
      assistantId,
      { input: { messages: [{ role: "human", content: "what's the weather in sf?" }] } }
    );

    let run = await client.runs.create(
      thread["thread_id"],
      assistant_id,
      { 
        input: { messages: [{ role: "human", content: "what's the weather in nyc?" }] },
        multitaskStrategy: "rollback" 
      }
    );

    // wait until the second run completes
    await client.runs.join(thread["thread_id"], run["run_id"]);
    ```

=== "CURL"

    ```bash
    curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in sf?\"}]},
    }" && curl --request POST \
    --url <DEPLOY<ENT_URL>>/threads/<THREAD_ID>/runs \
    --header 'Content-Type: application/json' \
    --data "{
      \"assistant_id\": \"agent\",
      \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what\'s the weather in nyc?\"}]},
      \"multitask_strategy\": \"rollback\"
    }" && curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/<RUN_ID>/join
    ```

### View run results

We can see that the thread has data only from the second run

=== "Python"

    ```python
    state = await client.threads.get_state(thread["thread_id"])

    for m in convert_to_messages(state["values"]["messages"]):
        m.pretty_print()
    ```

=== "Javascript"

    ```js
    const state = await client.threads.getState(thread["thread_id"]);

    for (const m of state['values']['messages']) {
      prettyPrint(m);
    }
    ```

=== "CURL"

    ```bash
    source pretty_print.sh && curl --request GET \
    --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/state | \
    jq -c '.values.messages[]' | while read -r element; do
        type=$(echo "$element" | jq -r '.type')
        content=$(echo "$element" | jq -r '.content | if type == "array" then tostring else . end')
        pretty_print "$type" "$content"
    done
    ```

Output:

    ================================ Human Message =================================
    
    what's the weather in nyc?
    ================================== Ai Message ==================================
    
    [{'id': 'toolu_01JzPqefao1gxwajHQ3Yh3JD', 'input': {'query': 'weather in nyc'}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
    Tool Calls:
      tavily_search_results_json (toolu_01JzPqefao1gxwajHQ3Yh3JD)
     Call ID: toolu_01JzPqefao1gxwajHQ3Yh3JD
      Args:
        query: weather in nyc
    ================================= Tool Message =================================
    Name: tavily_search_results_json
    
    [{"url": "https://www.weatherapi.com/", "content": "{'location': {'name': 'New York', 'region': 'New York', 'country': 'United States of America', 'lat': 40.71, 'lon': -74.01, 'tz_id': 'America/New_York', 'localtime_epoch': 1718734479, 'localtime': '2024-06-18 14:14'}, 'current': {'last_updated_epoch': 1718733600, 'last_updated': '2024-06-18 14:00', 'temp_c': 29.4, 'temp_f': 84.9, 'is_day': 1, 'condition': {'text': 'Sunny', 'icon': '//cdn.weatherapi.com/weather/64x64/day/113.png', 'code': 1000}, 'wind_mph': 2.2, 'wind_kph': 3.6, 'wind_degree': 158, 'wind_dir': 'SSE', 'pressure_mb': 1025.0, 'pressure_in': 30.26, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 63, 'cloud': 0, 'feelslike_c': 31.3, 'feelslike_f': 88.3, 'windchill_c': 28.3, 'windchill_f': 82.9, 'heatindex_c': 29.6, 'heatindex_f': 85.3, 'dewpoint_c': 18.4, 'dewpoint_f': 65.2, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 7.0, 'gust_mph': 16.5, 'gust_kph': 26.5}}"}]
    ================================== Ai Message ==================================
    
    The weather API results show that the current weather in New York City is sunny with a temperature of around 85┬░F (29┬░C). The wind is light at around 2-3 mph from the south-southeast. Overall it looks like a nice sunny summer day in NYC.


Verify that the original, rolled back run was deleted

=== "Python"

    ```python
    try:
        await client.runs.get(thread["thread_id"], rolled_back_run["run_id"])
    except httpx.HTTPStatusError as _:
        print("Original run was correctly deleted")
    ```

=== "Javascript"

    ```js
    try {
      await client.runs.get(thread["thread_id"], rolledBackRun["run_id"]);
    } catch (e) {
      console.log("Original run was correctly deleted");
    }
    ```

Output:

    Original run was correctly deleted



<a name="docscloudhow-tossame-threadmd"></a>

## How to run multiple agents on the same thread

In LangGraph Cloud, a thread is not explicitly associated with a particular agent.
This means that you can run multiple agents on the same thread, which allows a different agent to continue from an initial agent's progress.

In this example, we will create two agents and then call them both on the same thread.
You'll see that the second agent will respond using information from the [checkpoint](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer-state) generated in the thread by the first agent as context.

### Setup

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)

    openai_assistant = await client.assistants.create(
        graph_id="agent", config={"configurable": {"model_name": "openai"}}
    )

    # There should always be a default assistant with no configuration
    assistants = await client.assistants.search()
    default_assistant = [a for a in assistants if not a["config"]][0]
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    
    const openAIAssistant = await client.assistants.create(
      { graphId: "agent", config: {"configurable": {"model_name": "openai"}}}
    );

    const assistants = await client.assistants.search();
    const defaultAssistant = assistants.find(a => !a.config);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants \
        --header 'Content-Type: application/json' \
        --data '{
            "graph_id": "agent",
            "config": { "configurable": { "model_name": "openai" } }
        }' && \
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0]'
    ```

We can see that these agents are different:

=== "Python"

    ```python
    print(openai_assistant)
    ```

=== "Javascript"

    ```js
    console.log(openAIAssistant);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/assistants/<OPENAI_ASSISTANT_ID>
    ```

Output:

    {
        "assistant_id": "db87f39d-b2b1-4da8-ac65-cf81beb3c766",
        "graph_id": "agent",
        "created_at": "2024-08-30T21:18:51.850581+00:00",
        "updated_at": "2024-08-30T21:18:51.850581+00:00",
        "config": {
            "configurable": {
                "model_name": "openai"
            }
        },
        "metadata": {}
    }

=== "Python"

    ```python
    print(default_assistant)
    ```

=== "Javascript"

    ```js
    console.log(defaultAssistant);
    ```

=== "CURL"

    ```bash
    curl --request GET \
        --url <DEPLOYMENT_URL>/assistants/<DEFAULT_ASSISTANT_ID>
    ```

Output:

    {
        "assistant_id": "fe096781-5601-53d2-b2f6-0d3403f7e9ca",
        "graph_id": "agent",
        "created_at": "2024-08-08T22:45:24.562906+00:00",
        "updated_at": "2024-08-08T22:45:24.562906+00:00",
        "config": {},
        "metadata": {
            "created_by": "system"
        }
    }

### Run assistants on thread

#### Run OpenAI assistant

We can now run the OpenAI assistant on the thread first.

=== "Python"

    ```python
    thread = await client.threads.create()
    input = {"messages": [{"role": "user", "content": "who made you?"}]}
    async for event in client.runs.stream(
        thread["thread_id"],
        openai_assistant["assistant_id"],
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving event of type: {event.event}")
        print(event.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const thread = await client.threads.create();
    let input =  {"messages": [{"role": "user", "content": "who made you?"}]}

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      openAIAssistant["assistant_id"],
      {
        input,
        streamMode: "updates"
      }
    );
    for await (const event of streamResponse) {
      console.log(`Receiving event of type: ${event.event}`);
      console.log(event.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    thread_id=$(curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}' | jq -r '.thread_id') && \
    curl --request POST \
        --url "<DEPLOYMENT_URL>/threads/${thread_id}/runs/stream" \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <OPENAI_ASSISTANT_ID>,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "who made you?"
                    }
                ]
            },
            "stream_mode": [
                "updates"
            ]
        }' | \
        sed 's/\r$//' | \
        awk '
        /^event:/ {
            if (data_content != "") {
                print data_content "\n"
            }
            sub(/^event: /, "Receiving event of type: ", $0)
            printf "%s...\n", $0
            data_content = ""
        }
        /^data:/ {
            sub(/^data: /, "", $0)
            data_content = $0
        }
        END {
            if (data_content != "") {
                print data_content "\n\n"
            }
        }
    '
    ```

Output:

    Receiving event of type: metadata
    {'run_id': '1ef671c5-fb83-6e70-b698-44dba2d9213e'}


    Receiving event of type: updates
    {'agent': {'messages': [{'content': 'I was created by OpenAI, a research organization focused on developing and advancing artificial intelligence technology.', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-f5735b86-b80d-4c71-8dc3-4782b5a9c7c8', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}

#### Run default assistant

Now, we can run it on the default assistant and see that this second assistant is aware of the initial question, and can answer the question, "and you?":

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "and you?"}]}
    async for event in client.runs.stream(
        thread["thread_id"],
        default_assistant["assistant_id"],
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving event of type: {event.event}")
        print(event.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    let input =  {"messages": [{"role": "user", "content": "and you?"}]}

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      defaultAssistant["assistant_id"],
      {
        input,
        streamMode: "updates"
      }
    );
    for await (const event of streamResponse) {
      console.log(`Receiving event of type: ${event.event}`);
      console.log(event.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <DEFAULT_ASSISTANT_ID>,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "and you?"
                    }
                ]
            },
            "stream_mode": [
                "updates"
            ]
        }' | \
        sed 's/\r$//' | \
        awk '
        /^event:/ {
            if (data_content != "") {
                print data_content "\n"
            }
            sub(/^event: /, "Receiving event of type: ", $0)
            printf "%s...\n", $0
            data_content = ""
        }
        /^data:/ {
            sub(/^data: /, "", $0)
            data_content = $0
        }
        END {
            if (data_content != "") {
                print data_content "\n\n"
            }
        }
    '
    ```

Output:

    Receiving event of type: metadata
    {'run_id': '1ef6722d-80b3-6fbb-9324-253796b1cd13'}


    Receiving event of type: updates
    {'agent': {'messages': [{'content': [{'text': 'I am an artificial intelligence created by Anthropic, not by OpenAI. I should not have stated that OpenAI created me, as that is incorrect. Anthropic is the company that developed and trained me using advanced language models and AI technology. I will be more careful about providing accurate information regarding my origins in the future.', 'type': 'text', 'index': 0}], 'additional_kwargs': {}, 'response_metadata': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'type': 'ai', 'name': None, 'id': 'run-ebaacf62-9dd9-4165-9535-db432e4793ec', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 302, 'output_tokens': 72, 'total_tokens': 374}}]}}





<a name="docscloudhow-tosstateless_runsmd"></a>

## Stateless Runs

Most of the time, you provide a `thread_id` to your client when you run your graph in order to keep track of prior runs through the persistent state implemented in LangGraph Cloud. However, if you don't need to persist the runs you don't need to use the built in persistent state and can create stateless runs.

### Setup

First, let's setup our client:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantId = "agent";
    // create thread
    const thread = await client.threads.create();
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0].graph_id' && \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

### Stateless streaming

We can stream the results of a stateless run in an almost identical fashion to how we stream from a run with the state attribute, but instead of passing a value to the `thread_id` parameter, we pass `None`:

=== "Python"

    ```python
    input = {
        "messages": [
            {"role": "user", "content": "Hello! My name is Bagatur and I am 26 years old."}
        ]
    }

    async for chunk in client.runs.stream(
        # Don't pass in a thread_id and the stream will be stateless
        None,
        assistant_id,
        input=input,
        stream_mode="updates",
    ):
        if chunk.data and "run_id" not in chunk.data:
            print(chunk.data)
    ```

=== "Javascript"

    ```js
    let input = {
      messages: [
        { role: "user", content: "Hello! My name is Bagatur and I am 26 years old." }
      ]
    };

    const streamResponse = client.runs.stream(
      // Don't pass in a thread_id and the stream will be stateless
      null,
      assistantId,
      {
        input,
        streamMode: "updates"
      }
    );
    for await (const chunk of streamResponse) {
      if (chunk.data && !("run_id" in chunk.data)) {
        console.log(chunk.data);
      }
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data "{
            \"assistant_id\": \"agent\",
            \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"Hello! My name is Bagatur and I am 26 years old.\"}]},
            \"stream_mode\": [
                \"updates\"
            ]
        }" | jq -c 'select(.data and (.data | has("run_id") | not)) | .data'
    ```

Output:

    {'agent': {'messages': [{'content': "Hello Bagatur! It's nice to meet you. Thank you for introducing yourself and sharing your age. Is there anything specific you'd like to know or discuss? I'm here to help with any questions or topics you're interested in.", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-489ec573-1645-4ce2-a3b8-91b391d50a71', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}

### Waiting for stateless results

In addition to streaming, you can also wait for a stateless result by using the `.wait` function like follows:

=== "Python"

    ```python
    stateless_run_result = await client.runs.wait(
        None,
        assistant_id,
        input=input,
    )
    print(stateless_run_result)
    ```

=== "Javascript"

    ```js
    let statelessRunResult = await client.runs.wait(
      null,
      assistantId,
      { input: input }
    );
    console.log(statelessRunResult);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/runs/runs/wait \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_IDD>,
        }'
    ```

Output:

    {
        'messages': [
            {
                'content': 'Hello! My name is Bagatur and I am 26 years old.',
                'additional_kwargs': {},
                'response_metadata': {},
                'type': 'human',
                'name': None,
                'id': '5e088543-62c2-43de-9d95-6086ad7f8b48',
                'example': False}
            ,
            {
                'content': "Hello Bagatur! It's nice to meet you. Thank you for introducing yourself and sharing your age. Is there anything specific you'd like to know or discuss? I'm here to help with any questions or topics you'd like to explore.",
                'additional_kwargs': {},
                'response_metadata': {},
                'type': 'ai',
                'name': None,
                'id': 'run-d6361e8d-4d4c-45bd-ba47-39520257f773',
                'example': False,
                'tool_calls': [],
                'invalid_tool_calls': [],
                'usage_metadata': None
            }
        ]
    }

<a name="docscloudhow-tosstream_debugmd"></a>

## How to stream debug events

!!! info "Prerequisites"
    * [Streaming](#docsconceptsstreamingmd)

This guide covers how to stream debug events from your graph (`stream_mode="debug"`). Streaming debug events produces responses containing `type` and `timestamp` keys. Debug events correspond to different steps in the graph's execution, and there are three different types of steps that will get streamed back to you:

- `checkpoint`: These events will get streamed anytime the graph saves its state, which occurs after every super-step. Read more about checkpoints [here](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer)
- `task`: These events will get streamed before each super-step, and will contain information about a single task. Each super-step works by executing a list of tasks, where each task is scoped to a specific node and input. Below we will discuss the format of these tasks in more detail. 
- `task_result`: After each `task` event, you will see a corresponding `task_result` event which as the name suggests contains information on the results of the task executed in the super-step. Scroll more to learn about the exact structure of these events.

### Setup

First let's set up our client and thread:

=== "Python"
    
    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```


Output:

    {
        'thread_id': 'd0cbe9ad-f11c-443a-9f6f-dca0ae5a0dd3',
        'created_at': '2024-06-21T22:10:27.696862+00:00',
        'updated_at': '2024-06-21T22:10:27.696862+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

### Stream graph in debug mode

=== "Python"

    ```python
    # create input
    input = {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in SF?",
            }
        ]
    }

    # stream debug
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        stream_mode="debug",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    // create input
    const input = {
      messages: [
        {
          role: "human",
          content: "What's the weather in SF?",
        }
      ]
    };

    // stream debug
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: "debug"
      }
    );

    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in SF?\"}]},
       \"stream_mode\": [
         \"debug\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "") {
             print data_content "\n"
         }
         sub(/^event: /, "Receiving event of type: ", $0)
         printf "%s...\n", $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "") {
             print data_content "\n"
         }
     }
     ' 
    ```


Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2'}



    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-08-28T23:16:28.134680+00:00', 'step': -1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'checkpoint_id': '1ef65938-d8f3-6b25-bfff-30a8ed6460bd', 'checkpoint_ns': ''}, 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2'}, 'values': {'messages': [], 'search_results': []}, 'metadata': {'source': 'input', 'writes': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}, 'step': -1}, 'next': ['__start__'], 'tasks': [{'id': 'b40d2c90-dc1e-52db-82d6-08751b769c55', 'name': '__start__', 'interrupts': []}]}}



    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-08-28T23:16:28.139821+00:00', 'step': 0, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'checkpoint_id': '1ef65938-d900-63f1-8000-70fe53e0da5c', 'checkpoint_ns': ''}, 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4123a12c-46cb-4815-bdcc-32537af0cb5b', 'example': False}], 'search_results': []}, 'metadata': {'source': 'loop', 'writes': None, 'step': 0}, 'next': ['call_model'], 'tasks': [{'id': '685d89f6-542b-5e11-8cff-2963e7f4ea63', 'name': 'call_model', 'interrupts': []}]}}



    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-08-28T23:16:28.139928+00:00', 'step': 1, 'payload': {'id': '600a6ff3-7ff1-570a-b626-f887e9a70f1c', 'name': 'call_model', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4123a12c-46cb-4815-bdcc-32537af0cb5b', 'example': False}], 'search_results': [], 'final_answer': None}, 'triggers': ['start:call_model']}}



    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-08-28T23:16:28.584833+00:00', 'step': 1, 'payload': {'id': '600a6ff3-7ff1-570a-b626-f887e9a70f1c', 'name': 'call_model', 'error': None, 'result': [['messages', {'content': 'Current weather in San Francisco', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5'}, 'type': 'ai', 'name': None, 'id': 'run-0407bff9-3692-4ab5-9e57-2e9f396a3ee4', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]], 'interrupts': []}}



    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-08-28T23:16:28.584991+00:00', 'step': 1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'checkpoint_id': '1ef65938-dd3f-616f-8001-ce1c6f31e130', 'checkpoint_ns': ''}, 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4123a12c-46cb-4815-bdcc-32537af0cb5b', 'example': False}, {'content': 'Current weather in San Francisco', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5'}, 'type': 'ai', 'name': None, 'id': 'run-0407bff9-3692-4ab5-9e57-2e9f396a3ee4', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'search_results': []}, 'metadata': {'source': 'loop', 'writes': {'call_model': {'messages': {'content': 'Current weather in San Francisco', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5'}, 'type': 'ai', 'name': None, 'id': 'run-0407bff9-3692-4ab5-9e57-2e9f396a3ee4', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}}}, 'step': 1}, 'next': ['exa_search', 'tavily_search'], 'tasks': [{'id': '43865935-be38-5f6e-8d38-d44ef369c278', 'name': 'exa_search', 'interrupts': []}, {'id': 'dc220677-2720-56c7-a524-caaff60fce2c', 'name': 'tavily_search', 'interrupts': []}]}}



    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-08-28T23:16:28.585219+00:00', 'step': 2, 'payload': {'id': '870b5854-2f84-533d-8e7d-87158ee948fc', 'name': 'exa_search', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4123a12c-46cb-4815-bdcc-32537af0cb5b', 'example': False}, {'content': 'Current weather in San Francisco', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5'}, 'type': 'ai', 'name': None, 'id': 'run-0407bff9-3692-4ab5-9e57-2e9f396a3ee4', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'search_results': [], 'final_answer': None}, 'triggers': ['call_model']}}



    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-08-28T23:16:28.585219+00:00', 'step': 2, 'payload': {'id': '7589abfc-04df-58c6-8835-be172f84a7ff', 'name': 'tavily_search', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4123a12c-46cb-4815-bdcc-32537af0cb5b', 'example': False}, {'content': 'Current weather in San Francisco', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5'}, 'type': 'ai', 'name': None, 'id': 'run-0407bff9-3692-4ab5-9e57-2e9f396a3ee4', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'search_results': [], 'final_answer': None}, 'triggers': ['call_model']}}



    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-08-28T23:16:32.422243+00:00', 'step': 2, 'payload': {'id': '7589abfc-04df-58c6-8835-be172f84a7ff', 'name': 'tavily_search', 'error': None, 'result': [['search_results', ["{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1724886988, 'localtime': '2024-08-28 16:16'}, 'current': {'last_updated_epoch': 1724886900, 'last_updated': '2024-08-28 16:15', 'temp_c': 22.2, 'temp_f': 72.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 16.1, 'wind_kph': 25.9, 'wind_degree': 300, 'wind_dir': 'WNW', 'pressure_mb': 1013.0, 'pressure_in': 29.91, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 61, 'cloud': 25, 'feelslike_c': 24.6, 'feelslike_f': 76.4, 'windchill_c': 19.6, 'windchill_f': 67.2, 'heatindex_c': 19.7, 'heatindex_f': 67.4, 'dewpoint_c': 13.0, 'dewpoint_f': 55.5, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 18.7, 'gust_kph': 30.0}}"]]], 'interrupts': []}}



    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-08-28T23:16:34.750124+00:00', 'step': 2, 'payload': {'id': '870b5854-2f84-533d-8e7d-87158ee948fc', 'name': 'exa_search', 'error': None, 'result': [['search_results', ['The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.']]], 'interrupts': []}}



    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-08-28T23:16:34.750266+00:00', 'step': 2, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'checkpoint_id': '1ef65939-180b-6087-8002-f969296f8e3d', 'checkpoint_ns': ''}, 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4123a12c-46cb-4815-bdcc-32537af0cb5b', 'example': False}, {'content': 'Current weather in San Francisco', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5'}, 'type': 'ai', 'name': None, 'id': 'run-0407bff9-3692-4ab5-9e57-2e9f396a3ee4', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'search_results': ['The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.', "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1724886988, 'localtime': '2024-08-28 16:16'}, 'current': {'last_updated_epoch': 1724886900, 'last_updated': '2024-08-28 16:15', 'temp_c': 22.2, 'temp_f': 72.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 16.1, 'wind_kph': 25.9, 'wind_degree': 300, 'wind_dir': 'WNW', 'pressure_mb': 1013.0, 'pressure_in': 29.91, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 61, 'cloud': 25, 'feelslike_c': 24.6, 'feelslike_f': 76.4, 'windchill_c': 19.6, 'windchill_f': 67.2, 'heatindex_c': 19.7, 'heatindex_f': 67.4, 'dewpoint_c': 13.0, 'dewpoint_f': 55.5, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 18.7, 'gust_kph': 30.0}}"]}, 'metadata': {'source': 'loop', 'writes': {'exa_search': {'search_results': ['The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.']}, 'tavily_search': {'search_results': ["{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1724886988, 'localtime': '2024-08-28 16:16'}, 'current': {'last_updated_epoch': 1724886900, 'last_updated': '2024-08-28 16:15', 'temp_c': 22.2, 'temp_f': 72.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 16.1, 'wind_kph': 25.9, 'wind_degree': 300, 'wind_dir': 'WNW', 'pressure_mb': 1013.0, 'pressure_in': 29.91, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 61, 'cloud': 25, 'feelslike_c': 24.6, 'feelslike_f': 76.4, 'windchill_c': 19.6, 'windchill_f': 67.2, 'heatindex_c': 19.7, 'heatindex_f': 67.4, 'dewpoint_c': 13.0, 'dewpoint_f': 55.5, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 18.7, 'gust_kph': 30.0}}"]}}, 'step': 2}, 'next': ['summarize_search_results'], 'tasks': [{'id': '7263c738-516d-5708-b318-2c8ef54d4a33', 'name': 'summarize_search_results', 'interrupts': []}]}}



    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-08-28T23:16:34.750394+00:00', 'step': 3, 'payload': {'id': '5beaa05d-57d4-5acd-95c1-c7093990910f', 'name': 'summarize_search_results', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4123a12c-46cb-4815-bdcc-32537af0cb5b', 'example': False}, {'content': 'Current weather in San Francisco', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5'}, 'type': 'ai', 'name': None, 'id': 'run-0407bff9-3692-4ab5-9e57-2e9f396a3ee4', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'search_results': ['The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.', "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1724886988, 'localtime': '2024-08-28 16:16'}, 'current': {'last_updated_epoch': 1724886900, 'last_updated': '2024-08-28 16:15', 'temp_c': 22.2, 'temp_f': 72.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 16.1, 'wind_kph': 25.9, 'wind_degree': 300, 'wind_dir': 'WNW', 'pressure_mb': 1013.0, 'pressure_in': 29.91, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 61, 'cloud': 25, 'feelslike_c': 24.6, 'feelslike_f': 76.4, 'windchill_c': 19.6, 'windchill_f': 67.2, 'heatindex_c': 19.7, 'heatindex_f': 67.4, 'dewpoint_c': 13.0, 'dewpoint_f': 55.5, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 18.7, 'gust_kph': 30.0}}"], 'final_answer': None}, 'triggers': ['exa_search', 'tavily_search']}}



    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-08-28T23:16:35.851058+00:00', 'step': 3, 'payload': {'id': '5beaa05d-57d4-5acd-95c1-c7093990910f', 'name': 'summarize_search_results', 'error': None, 'result': [['final_answer', {'content': "The provided data details various twilight periods based on the sun's position relative to the horizon, alongside current weather information for San Francisco, California, as of August 28, 2024. The weather is partly cloudy with a temperature of 22.2┬░C (72.0┬░F), moderate wind from the WNW at 16.1 mph, and the UV index is 5.", 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-928c997b-9d85-4664-bd20-97ade4cc655e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]], 'interrupts': []}}



    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-08-28T23:16:35.851194+00:00', 'step': 3, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'be4fd54d-ff22-4e9e-8876-d5cccc0e8048', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'checkpoint_id': '1ef65939-228a-6d93-8003-8b06d7483024', 'checkpoint_ns': ''}, 'run_id': '1ef65938-d7c7-68db-b786-011aa1cb3cd2'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '4123a12c-46cb-4815-bdcc-32537af0cb5b', 'example': False}, {'content': 'Current weather in San Francisco', 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_a2ff031fb5'}, 'type': 'ai', 'name': None, 'id': 'run-0407bff9-3692-4ab5-9e57-2e9f396a3ee4', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'search_results': ['The time period when the sun is no more than 6 degrees below the horizon at either sunrise or sunset. The horizon should be clearly defined and the brightest stars should be visible under good atmospheric conditions (i.e. no moonlight, or other lights). One still should be able to carry on ordinary outdoor activities. The time period when the sun is between 6 and 12 degrees below the horizon at either sunrise or sunset. The horizon is well defined and the outline of objects might be visible without artificial light. Ordinary outdoor activities are not possible at this time without extra illumination. The time period when the sun is between 12 and 18 degrees below the horizon at either sunrise or sunset. The sun does not contribute to the illumination of the sky before this time in the morning, or after this time in the evening. In the beginning of morning astronomical twilight and at the end of astronomical twilight in the evening, sky illumination is very faint, and might be undetectable. The time of Civil Sunset minus the time of Civil Sunrise. The time of Actual Sunset minus the time of Actual Sunrise. The change in length of daylight between today and tomorrow is also listed when available.', "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.78, 'lon': -122.42, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1724886988, 'localtime': '2024-08-28 16:16'}, 'current': {'last_updated_epoch': 1724886900, 'last_updated': '2024-08-28 16:15', 'temp_c': 22.2, 'temp_f': 72.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 16.1, 'wind_kph': 25.9, 'wind_degree': 300, 'wind_dir': 'WNW', 'pressure_mb': 1013.0, 'pressure_in': 29.91, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 61, 'cloud': 25, 'feelslike_c': 24.6, 'feelslike_f': 76.4, 'windchill_c': 19.6, 'windchill_f': 67.2, 'heatindex_c': 19.7, 'heatindex_f': 67.4, 'dewpoint_c': 13.0, 'dewpoint_f': 55.5, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 5.0, 'gust_mph': 18.7, 'gust_kph': 30.0}}"], 'final_answer': {'content': "The provided data details various twilight periods based on the sun's position relative to the horizon, alongside current weather information for San Francisco, California, as of August 28, 2024. The weather is partly cloudy with a temperature of 22.2┬░C (72.0┬░F), moderate wind from the WNW at 16.1 mph, and the UV index is 5.", 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-928c997b-9d85-4664-bd20-97ade4cc655e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}}, 'metadata': {'source': 'loop', 'writes': {'summarize_search_results': {'final_answer': {'content': "The provided data details various twilight periods based on the sun's position relative to the horizon, alongside current weather information for San Francisco, California, as of August 28, 2024. The weather is partly cloudy with a temperature of 22.2┬░C (72.0┬░F), moderate wind from the WNW at 16.1 mph, and the UV index is 5.", 'additional_kwargs': {}, 'response_metadata': {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-05-13', 'system_fingerprint': 'fp_157b3831f5'}, 'type': 'ai', 'name': None, 'id': 'run-928c997b-9d85-4664-bd20-97ade4cc655e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}}}, 'step': 3}, 'next': [], 'tasks': []}}



We see that our debug events start with two `checkpoint` events at step 0 and 1, which represent checkpointing before the graph is created and after it has been created. We then see a single `task` and corresponding `task_result` which corresponds to our first node, `call_model`, being triggered. After it has finished, the entire super-step is over so the graph saves another checkpoint and we see the corresponding `checkpoint` event. 

The next super-step executed two search nodes [in parallel](https://langchain-ai.github.io/langgraph/how-tos/branching/) - specifically one node will execute an Exa search, while the other will use Tavily. Executing these nodes in parallel in the same super-step creates 2 `task` events and two corresponding `task_result` events. After we receive both of those `task_result` events, we see another `checkpoint` event as we would expect.

Lastly, we see a final `task` and `task_result` pair corresponding to the `summarize_search_results` node, which is the last node in our graph. As soon as this super-step is done we see one final `checkpoint` event corresponding to the final checkpoint of this run.
    



<a name="docscloudhow-tosstream_eventsmd"></a>

## How to stream events

!!! info "Prerequisites"
    * [Streaming](#streaming-llm-tokens-and-events-astream_events)

This guide covers how to stream events from your graph (`stream_mode="events"`). Depending on the use case and user experience of your LangGraph application, your application may process event types differently.

### Setup

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

Output:


    {
        'thread_id': '3f4c64e0-f792-4a5e-aa07-a4404e06e0bd',
        'created_at': '2024-06-24T22:16:29.301522+00:00',
        'updated_at': '2024-06-24T22:16:29.301522+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

### Stream graph in events mode

Streaming events produces responses containing an `event` key (in addition to other keys such as `data`). See the LangChain [`Runnable.astream_events()` reference](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) for all event types.


=== "Python"

    ```python
    # create input
    input = {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in SF?",
            }
        ]
    }

    # stream events
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        stream_mode="events",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    // create input
    const input = {
      "messages": [
        {
          "role": "user",
          "content": "What's the weather in SF?",
        }
      ]
    }

    // stream events
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: "events"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in sf\"}]},
       \"stream_mode\": [
         \"events\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "") {
             print data_content "\n"
         }
         sub(/^event: /, "Receiving event of type: ", $0)
         printf "%s...\n", $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "") {
             print data_content "\n"
         }
     }
     ' 
    ```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8'}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}, 'name': 'LangGraph', 'tags': [], 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'agent', 'tags': ['graph:step:6'], 'run_id': '7bb08493-d507-4e28-b9e6-4a5eda9d04f0', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}]]}}, 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'b', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'g', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'i', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'n', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_end', 'data': {'output': {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, 'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}]]}}, 'run_id': 'cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'name': 'should_continue', 'tags': ['seq:step:3'], 'run_id': 'c7fe4d2d-3fb8-4e53-946d-03de13527853', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': 'tool', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'run_id': 'c7fe4d2d-3fb8-4e53-946d-03de13527853', 'name': 'should_continue', 'tags': ['seq:step:3'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '7bb08493-d507-4e28-b9e6-4a5eda9d04f0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '7bb08493-d507-4e28-b9e6-4a5eda9d04f0', 'name': 'agent', 'tags': ['graph:step:6'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}], 'sleep': None}}, 'run_id': '7bb08493-d507-4e28-b9e6-4a5eda9d04f0', 'name': 'agent', 'tags': ['graph:step:6'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 6, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'tool', 'tags': ['graph:step:7'], 'run_id': 'f044fd3d-7271-488f-b8aa-e01572ff9112', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 7, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': 'f044fd3d-7271-488f-b8aa-e01572ff9112', 'name': 'tool', 'tags': ['graph:step:7'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 7, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': None, 'tool_call_id': 'tool_call_id'}]}}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}]}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}}, 'run_id': 'f044fd3d-7271-488f-b8aa-e01572ff9112', 'name': 'tool', 'tags': ['graph:step:7'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 7, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'agent', 'tags': ['graph:step:8'], 'run_id': '1f4f95d0-0ce1-4061-85d4-946446bbd3e5', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}]]}}, 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'n', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'd', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_end', 'data': {'output': {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, 'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}]]}}, 'run_id': '028a68fb-6435-4b46-a156-c3326f73985c', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'name': 'should_continue', 'tags': ['seq:step:3'], 'run_id': 'f2b2dfaf-475d-422b-8bf5-02a31bcc7d1a', 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': '__end__', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'run_id': 'f2b2dfaf-475d-422b-8bf5-02a31bcc7d1a', 'name': 'should_continue', 'tags': ['seq:step:3'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8', '1f4f95d0-0ce1-4061-85d4-946446bbd3e5']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1f4f95d0-0ce1-4061-85d4-946446bbd3e5', 'name': 'agent', 'tags': ['graph:step:8'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}], 'sleep': None}}, 'run_id': '1f4f95d0-0ce1-4061-85d4-946446bbd3e5', 'name': 'agent', 'tags': ['graph:step:8'], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 8, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef301a5-b867-67de-9e9e-a32e53c5b1f8']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '51f2874d-f8c7-4040-8b3b-8f15429a56ae', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-5f556aa0-26ea-42e2-b9e4-7ece3a00974e', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1faf5dd0-ae97-4235-963f-5075083a027a', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-ae383611-6a42-475a-912a-09d5972e9e94', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': 'c67e08e6-e7af-4c4a-aa5e-50c8340ae341', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-cb1b98c1-c9e2-4a30-9d7a-38fa1f6224bd', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '1c9a16d2-5f0a-4eba-a0d2-240484a4ce7e', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-028a68fb-6435-4b46-a156-c3326f73985c', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}, 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'name': 'LangGraph', 'tags': [], 'metadata': {'graph_id': 'agent', 'created_by': 'system', 'run_id': '1ef301a5-b867-67de-9e9e-a32e53c5b1f8', 'user_id': '', 'thread_id': '7196a3aa-763c-4a8d-bfda-12fbfe1cd727', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'parent_ids': []}
    
    
    
    Receiving new event of type: end...
    None

<a name="docscloudhow-tosstream_messagesmd"></a>

## How to stream messages from your graph

!!! info "Prerequisites"
    * [Streaming](#docsconceptsstreamingmd)

This guide covers how to stream messages from your graph. With `stream_mode="messages-tuple"`, messages (i.e. individual LLM tokens) from any chat model invocations inside your graph nodes will be streamed back.

### Setup

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

Output:

    {
        'thread_id': 'e1431c95-e241-4d1d-a252-27eceb1e5c86',
        'created_at': '2024-06-21T15:48:59.808924+00:00',
        'updated_at': '2024-06-21T15:48:59.808924+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

### Stream graph in messages mode

Now we can stream LLM tokens for any messages generated inside a node in the form of tuples `(message, metadata)`. Metadata contains additional information that can be useful for filtering the streamed outputs to a specific node or LLM.

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in sf"}]}
    config = {"configurable": {"model_name": "openai"}}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        config=config,
        stream_mode="messages-tuple",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const input = {
      messages: [
        {
          role: "human",
          content: "What's the weather in sf",
        }
      ]
    };
    const config = { configurable: { model_name: "openai" } };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        config,
        streamMode: "messages-tuple"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in la\"}]},
       \"stream_mode\": [
         \"messages-tuple\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "") {
             print data_content "\n"
         }
         sub(/^event: /, "Receiving event of type: ", $0)
         printf "%s...\n", $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "") {
             print data_content "\n"
         }
     }
     '
    ```


Output:

    Receiving new event of type: metadata...
    {"run_id": "1ef971e0-9a84-6154-9047-247b4ce89c4d", "attempt": 1}

    ...

    Receiving new event of type: messages...
    [
      {
        "type": "AIMessageChunk",
        "tool_calls": [
          {
            "name": "tavily_search_results_json",
            "args": {
              "query": "weat"
            },
            "id": "toolu_0114XKXdNtHQEa3ozmY1uDdM",
            "type": "tool_call"
          }
        ],
        ...
      },
      {
        "graph_id": "agent",
        "langgraph_node": "agent",
        ...
      }
    ]



    Receiving new event of type: messages...
    [
      {
        "type": "AIMessageChunk",
        "tool_calls": [
          {
            "name": "tavily_search_results_json",
            "args": {
              "query": "her in san "
            },
            "id": "toolu_0114XKXdNtHQEa3ozmY1uDdM",
            "type": "tool_call"
          }
        ],
        ...
      },
      {
        "graph_id": "agent",
        "langgraph_node": "agent",
        ...
      }
    ]

    ...

    Receiving new event of type: messages...
    [
      {
        "type": "AIMessageChunk",
        "tool_calls": [
          {
            "name": "tavily_search_results_json",
            "args": {
              "query": "francisco"
            },
            "id": "toolu_0114XKXdNtHQEa3ozmY1uDdM",
            "type": "tool_call"
          }
        ],
        ...
      },
      {
        "graph_id": "agent",
        "langgraph_node": "agent",
        ...
      }
    ]

    ...

    Receiving new event of type: messages...
    [
      {
        "content": "[{\"url\": \"https://www.weatherapi.com/\", \"content\": \"{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1730475777, 'localtime': '2024-11-01 08:42'}, 'current': {'last_updated_epoch': 1730475000, 'last_updated': '2024-11-01 08:30', 'temp_c': 11.1, 'temp_f': 52.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 2.2, 'wind_kph': 3.6, 'wind_degree': 192, 'wind_dir': 'SSW', 'pressure_mb': 1018.0, 'pressure_in': 30.07, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 89, 'cloud': 75, 'feelslike_c': 11.5, 'feelslike_f': 52.6, 'windchill_c': 10.0, 'windchill_f': 50.1, 'heatindex_c': 10.4, 'heatindex_f': 50.7, 'dewpoint_c': 9.1, 'dewpoint_f': 48.5, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 3.0, 'gust_mph': 6.7, 'gust_kph': 10.8}}\"}]",
        "type": "tool",
        "tool_call_id": "toolu_0114XKXdNtHQEa3ozmY1uDdM",
        ...
      },
      {
        "graph_id": "agent",
        "langgraph_node": "action",
        ...
      }
    ]

    ...

    Receiving new event of type: messages...
    [
      {
        "content": [
          {
            "text": "\n\nThe search",
            "type": "text",
            "index": 0
          }
        ],
        "type": "AIMessageChunk",
        ...
      },
      {
        "graph_id": "agent",
        "langgraph_node": "agent",
        ...
      }
    ]



    Receiving new event of type: messages...
    [
      {
        "content": [
          {
            "text": " results provide",
            "type": "text",
            "index": 0
          }
        ],
        "type": "AIMessageChunk",
        ...
      },
      {
        "graph_id": "agent",
        "langgraph_node": "agent",
        ...
      }
    ]



    Receiving new event of type: messages...
    [
      {
        "content": [
          {
            "text": " the current weather conditions",
            "type": "text",
            "index": 0
          }
        ],
        "type": "AIMessageChunk",
        ...
      },
      {
        "graph_id": "agent",
        "langgraph_node": "agent",
        ...
      }
    ]



    Receiving new event of type: messages...
    [
      {
        "content": [
          {
            "text": " in San Francisco.",
            "type": "text",
            "index": 0
          }
        ],
        "type": "AIMessageChunk",
        ...
      },
      {
        "graph_id": "agent",
        "langgraph_node": "agent",
        ...
      }
    ]

    ...

<a name="docscloudhow-tosstream_multiplemd"></a>

## How to configure multiple streaming modes at the same time

!!! info "Prerequisites"
    * [Streaming](#docsconceptsstreamingmd)

This guide covers how to configure multiple streaming modes at the same time.

### Setup

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

Output:

    {
        'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4',
        'created_at': '2024-06-24T21:30:07.980789+00:00',
        'updated_at': '2024-06-24T21:30:07.980789+00:00',
        'metadata': {},
        'status': 'idle',
        'config': {},
        'values': None
    }

### Stream graph with multiple modes

When configuring multiple streaming modes for a run, responses for each respective mode will be produced. In the following example, note that a `list` of modes (`messages`, `events`, `debug`) is passed to the `stream_mode` parameter and the response contains `events`, `debug`, `messages/complete`, `messages/metadata`, and `messages/partial` event types.

=== "Python"

    ```python
    # create input
    input = {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in SF?",
            }
        ]
    }

    # stream events with multiple streaming modes
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        stream_mode=["messages", "events", "debug"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    // create input
    const input = {
      messages: [
        {
          role: "human",
          content: "What's the weather in SF?",
        }
      ]
    };

    // stream events with multiple streaming modes
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: ["messages", "events", "debug"]
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in SF?\"}]},
       \"stream_mode\": [
         \"messages\",
         \"events\",
         \"debug\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "") {
             print data_content "\n"
         }
         sub(/^event: /, "Receiving event of type: ", $0)
         printf "%s...\n", $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    Receiving new event of type: metadata...
    {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}, 'name': 'LangGraph', 'tags': [], 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.116009+00:00', 'step': -1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc7c-6daa-bfff-6b9027c1a50e', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'messages': []}, 'metadata': {'source': 'input', 'step': -1, 'writes': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.116009+00:00', 'step': -1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc7c-6daa-bfff-6b9027c1a50e', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'messages': []}, 'metadata': {'source': 'input', 'step': -1, 'writes': {'messages': [{'role': 'human', 'content': "What's the weather in SF?"}]}}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['values', {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.117924+00:00', 'step': 0, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc81-68c8-8000-4e18ae7d67a5', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]}, 'metadata': {'source': 'loop', 'step': 0, 'writes': None}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.117924+00:00', 'step': 0, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc81-68c8-8000-4e18ae7d67a5', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]}, 'metadata': {'source': 'loop', 'step': 0, 'writes': None}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-24T21:34:06.118042+00:00', 'step': 1, 'payload': {'id': '212ed9c2-a454-50c5-a202-12066bbbe7b8', 'name': 'agent', 'input': {'some_bytes': None, 'some_byte_array': None, 'dict_with_bytes': None, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}], 'sleep': None}, 'triggers': ['start:agent']}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task', 'timestamp': '2024-06-24T21:34:06.118042+00:00', 'step': 1, 'payload': {'id': '212ed9c2-a454-50c5-a202-12066bbbe7b8', 'name': 'agent', 'input': {'some_bytes': None, 'some_byte_array': None, 'dict_with_bytes': None, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}], 'sleep': None}, 'triggers': ['start:agent']}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'agent', 'tags': ['graph:step:1'], 'run_id': '72b74d24-5792-48da-a887-102100d6e2c0', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]]}}, 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'b', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/metadata...
    {'run-2424dd6d-5cf5-4244-8d98-357640ce6e12': {'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}}}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'b', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'be', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'g', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'beg', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'i', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'begi', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'n', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_end', 'data': {'output': {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, 'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}]]}}, 'run_id': '2424dd6d-5cf5-4244-8d98-357640ce6e12', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'name': 'should_continue', 'tags': ['seq:step:3'], 'run_id': '227afb0f-f909-4d54-a042-556ca6d98a69', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': 'tool', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'run_id': '227afb0f-f909-4d54-a042-556ca6d98a69', 'name': 'should_continue', 'tags': ['seq:step:3'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', '72b74d24-5792-48da-a887-102100d6e2c0']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '72b74d24-5792-48da-a887-102100d6e2c0', 'name': 'agent', 'tags': ['graph:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}, 'input': {'some_bytes': None, 'some_byte_array': None, 'dict_with_bytes': None, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}], 'sleep': None}}, 'run_id': '72b74d24-5792-48da-a887-102100d6e2c0', 'name': 'agent', 'tags': ['graph:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 1, 'langgraph_node': 'agent', 'langgraph_triggers': ['start:agent'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.124350+00:00', 'step': 1, 'payload': {'id': '212ed9c2-a454-50c5-a202-12066bbbe7b8', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.124350+00:00', 'step': 1, 'payload': {'id': '212ed9c2-a454-50c5-a202-12066bbbe7b8', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['values', {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.124510+00:00', 'step': 1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc91-6a34-8001-26353c117c25', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 1, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.124510+00:00', 'step': 1, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc91-6a34-8001-26353c117c25', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 1, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-24T21:34:06.124572+00:00', 'step': 2, 'payload': {'id': '44139125-a1be-57c2-9cb2-19eb62bbaf2f', 'name': 'tool', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}, 'triggers': ['branch:agent:should_continue:tool']}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task', 'timestamp': '2024-06-24T21:34:06.124572+00:00', 'step': 2, 'payload': {'id': '44139125-a1be-57c2-9cb2-19eb62bbaf2f', 'name': 'tool', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}, 'triggers': ['branch:agent:should_continue:tool']}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'tool', 'tags': ['graph:step:2'], 'run_id': '91575720-886e-485e-ae2d-d6817e5346bf', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 2, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '91575720-886e-485e-ae2d-d6817e5346bf', 'name': 'tool', 'tags': ['graph:step:2'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 2, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': None, 'tool_call_id': 'tool_call_id'}]}}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'sleep': None}}, 'run_id': '91575720-886e-485e-ae2d-d6817e5346bf', 'name': 'tool', 'tags': ['graph:step:2'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 2, 'langgraph_node': 'tool', 'langgraph_triggers': ['branch:agent:should_continue:tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.126828+00:00', 'step': 2, 'payload': {'id': '44139125-a1be-57c2-9cb2-19eb62bbaf2f', 'name': 'tool', 'result': [['messages', [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]]]}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.126828+00:00', 'step': 2, 'payload': {'id': '44139125-a1be-57c2-9cb2-19eb62bbaf2f', 'name': 'tool', 'result': [['messages', [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]]]}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['values', {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: messages/complete...
    [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]
    
    
    
    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.126966+00:00', 'step': 2, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc97-6a06-8002-8e9ffc1ea75a', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}, 'metadata': {'source': 'loop', 'step': 2, 'writes': {'tool': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}}}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.126966+00:00', 'step': 2, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bc97-6a06-8002-8e9ffc1ea75a', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}, 'metadata': {'source': 'loop', 'step': 2, 'writes': {'tool': {'messages': [{'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]}}}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task', 'timestamp': '2024-06-24T21:34:06.127034+00:00', 'step': 3, 'payload': {'id': 'f1ccf371-63b3-5268-a837-7f360a93c4ec', 'name': 'agent', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}], 'sleep': None}, 'triggers': ['tool']}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task', 'timestamp': '2024-06-24T21:34:06.127034+00:00', 'step': 3, 'payload': {'id': 'f1ccf371-63b3-5268-a837-7f360a93c4ec', 'name': 'agent', 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}], 'sleep': None}, 'triggers': ['tool']}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {}, 'name': 'agent', 'tags': ['graph:step:3'], 'run_id': 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]]}}, 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: messages/metadata...
    {'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575': {'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}}}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'e', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'n', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'en', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_stream', 'data': {'chunk': {'content': 'd', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'AIMessageChunk', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None, 'tool_call_chunks': []}}, 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: messages/partial...
    [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chat_model_end', 'data': {'output': {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, 'input': {'messages': [[{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}]]}}, 'run_id': '0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'name': 'FakeListChatModel', 'tags': ['seq:step:1'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0, 'ls_model_type': 'chat'}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_start', 'data': {'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'name': 'should_continue', 'tags': ['seq:step:3'], 'run_id': '8af814e9-8136-4aab-acbc-dffc5bcafdfd', 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': '__end__', 'input': {'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'run_id': '8af814e9-8136-4aab-acbc-dffc5bcafdfd', 'name': 'should_continue', 'tags': ['seq:step:3'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25', 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e', 'name': 'agent', 'tags': ['graph:step:3'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'data': {'chunk': {'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}], 'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}}, 'input': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}], 'sleep': None}}, 'run_id': 'b7d0900c-bfc2-43e4-b760-99bbc5bad84e', 'name': 'agent', 'tags': ['graph:step:3'], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca', 'langgraph_step': 3, 'langgraph_node': 'agent', 'langgraph_triggers': ['tool'], 'langgraph_task_idx': 0}, 'parent_ids': ['1ef32717-bc30-6cf2-8a26-33f63567bc25']}
    
    
    
    Receiving new event of type: debug...
    {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.133991+00:00', 'step': 3, 'payload': {'id': 'f1ccf371-63b3-5268-a837-7f360a93c4ec', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'task_result', 'timestamp': '2024-06-24T21:34:06.133991+00:00', 'step': 3, 'payload': {'id': 'f1ccf371-63b3-5268-a837-7f360a93c4ec', 'name': 'agent', 'result': [['some_bytes', 'c29tZV9ieXRlcw=='], ['some_byte_array', 'c29tZV9ieXRlX2FycmF5'], ['dict_with_bytes', {'more_bytes': 'bW9yZV9ieXRlcw=='}], ['messages', [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]]]}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['values', {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}]}, 'parent_ids': []}
    
    

    Receiving new event of type: debug...
    {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.134190+00:00', 'step': 3, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bca9-6418-8003-8d0d0b06845c', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 3, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_stream', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'data': {'chunk': ['debug', {'type': 'checkpoint', 'timestamp': '2024-06-24T21:34:06.134190+00:00', 'step': 3, 'payload': {'config': {'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'callbacks': [None], 'recursion_limit': 25, 'configurable': {'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'thread_ts': '1ef32717-bca9-6418-8003-8d0d0b06845c', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25'}, 'values': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}, 'metadata': {'source': 'loop', 'step': 3, 'writes': {'agent': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}}}}]}, 'parent_ids': []}
    
    
    
    Receiving new event of type: events...
    {'event': 'on_chain_end', 'data': {'output': {'some_bytes': 'c29tZV9ieXRlcw==', 'some_byte_array': 'c29tZV9ieXRlX2FycmF5', 'dict_with_bytes': {'more_bytes': 'bW9yZV9ieXRlcw=='}, 'messages': [{'content': "What's the weather in SF?", 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'human', 'name': None, 'id': '7da1bafa-f53c-4df8-ba63-8dd517140b9f', 'example': False}, {'content': 'begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-2424dd6d-5cf5-4244-8d98-357640ce6e12', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}, {'content': 'tool_call__begin', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'tool', 'name': None, 'id': '639ca779-403d-4915-a066-327e1f634c8b', 'tool_call_id': 'tool_call_id'}, {'content': 'end', 'additional_kwargs': {}, 'response_metadata': {}, 'type': 'ai', 'name': None, 'id': 'run-0f2ef0a1-0fc7-445c-9df4-55e8bb284575', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': None}]}}, 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'name': 'LangGraph', 'tags': [], 'metadata': {'created_by': 'system', 'run_id': '1ef32717-bc30-6cf2-8a26-33f63567bc25', 'user_id': '', 'graph_id': 'agent', 'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4', 'assistant_id': 'fe096781-5601-53d2-b2f6-0d3403f7e9ca'}, 'parent_ids': []}
    
    
    
    Receiving new event of type: end...
    None
    
    



<a name="docscloudhow-tosstream_updatesmd"></a>

## How to stream state updates of your graph

!!! info "Prerequisites"
    * [Streaming](#docsconceptsstreamingmd)

This guide covers how to use `stream_mode="updates"` for your graph, which will stream the updates to the graph state that are made after each node is executed. This differs from using `stream_mode="values"`: instead of streaming the entire value of the state at each superstep, it only streams the updates from each of the nodes that made an update to the state at that superstep.

### Setup

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

Output:

    {
      'thread_id': '979e3c89-a702-4882-87c2-7a59a250ce16',
      'created_at': '2024-06-21T15:22:07.453100+00:00',
      'updated_at': '2024-06-21T15:22:07.453100+00:00',
      'metadata': {},
      'status': 'idle',
      'config': {},
      'values': None 
    }

### Stream graph in updates mode

Now we can stream by updates, which outputs updates made to the state by each node after it has executed:


=== "Python"

    ```python
    input = {
        "messages": [
            {
                "role": "user",
                "content": "what's the weather in la"
            }
        ]
    }
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const input = {
      messages: [
        {
          role: "human",
          content: "What's the weather in la"
        }
      ]
    };

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: "updates"
      }
    );

    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"What's the weather in la\"}]},
       \"stream_mode\": [
         \"updates\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "") {
             print data_content "\n"
         }
         sub(/^event: /, "Receiving event of type: ", $0)
         printf "%s...\n", $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "") {
             print data_content "\n"
         }
     }
     '
    ```

Output:

    Receiving new event of type: metadata...
    {"run_id": "cfc96c16-ed9a-44bd-b5bb-c30e3c0725f0"}



    Receiving new event of type: updates...
    {
      "agent": {
        "messages": [
          {
            "type": "ai",
            "tool_calls": [
              {
                "name": "tavily_search_results_json",
                "args": {
                  "query": "weather in los angeles"
                },
                "id": "toolu_0148tMmDK51iLQfG1yaNwRHM"
              }
            ],
            ...
          }
        ]
      }
    }



    Receiving new event of type: updates...
    {
      "action": {
        "messages": [
          {
            "content": [
              {
                "url": "https://www.weatherapi.com/",
                "content": "{\"location\": {\"name\": \"Los Angeles\", \"region\": \"California\", \"country\": \"United States of America\", \"lat\": 34.05, \"lon\": -118.24, \"tz_id\": \"America/Los_Angeles\", \"localtime_epoch\": 1716062239, \"localtime\": \"2024-05-18 12:57\"}, \"current\": {\"last_updated_epoch\": 1716061500, \"last_updated\": \"2024-05-18 12:45\", \"temp_c\": 18.9, \"temp_f\": 66.0, \"is_day\": 1, \"condition\": {\"text\": \"Overcast\", \"icon\": \"//cdn.weatherapi.com/weather/64x64/day/122.png\", \"code\": 1009}, \"wind_mph\": 2.2, \"wind_kph\": 3.6, \"wind_degree\": 10, \"wind_dir\": \"N\", \"pressure_mb\": 1017.0, \"pressure_in\": 30.02, \"precip_mm\": 0.0, \"precip_in\": 0.0, \"humidity\": 65, \"cloud\": 100, \"feelslike_c\": 18.9, \"feelslike_f\": 66.0, \"vis_km\": 16.0, \"vis_miles\": 9.0, \"uv\": 6.0, \"gust_mph\": 7.5, \"gust_kph\": 12.0}}"
              }
            ],
            "type": "tool",
            "name": "tavily_search_results_json",
            "tool_call_id": "toolu_0148tMmDK51iLQfG1yaNwRHM",
            ...
          }
        ]
      }
    }



    Receiving new event of type: updates...
    {
      "agent": {
        "messages": [
          {
            "content": "The weather in Los Angeles is currently overcast with a temperature of around 66┬░F (18.9┬░C). There are light winds from the north at around 2-3 mph. The humidity is 65% and visibility is good at 9 miles. Overall, mild spring weather conditions in LA.",
            "type": "ai",
            ...
          }
        ]
      }
    }



    Receiving new event of type: end...
    None

<a name="docscloudhow-tosstream_valuesmd"></a>

## How to stream full state of your graph

!!! info "Prerequisites"
    * [Streaming](#docsconceptsstreamingmd)

This guide covers how to use `stream_mode="values"`, which streams the value of the state at each superstep. This differs from using `stream_mode="updates"`: instead of streaming just the updates to the state from each node, it streams the entire graph state at that superstep.

### Setup

First let's set up our client and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
      --url <DEPLOYMENT_URL>/threads \
      --header 'Content-Type: application/json' \
      --data '{}'
    ```

Output:

    {
      'thread_id': 'bfc68029-1f7b-400f-beab-6f9032a52da4',
      'created_at': '2024-06-24T21:30:07.980789+00:00',
      'updated_at': '2024-06-24T21:30:07.980789+00:00',
      'metadata': {},
      'status': 'idle',
      'config': {},
      'values': None
    }

### Stream graph in values mode

Now we can stream by values, which streams the full state of the graph after each node has finished executing:

=== "Python"

    ```python
    input = {"messages": [{"role": "user", "content": "what's the weather in la"}]}

    # stream values
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id, 
        input=input,
        stream_mode="values"
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript"

    ```js
    const input = {"messages": [{"role": "user", "content": "what's the weather in la"}]}

    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: "values"
      }
    );
    for await (const chunk of streamResponse) {
      console.log(`Receiving new event of type: ${chunk.event}...`);
      console.log(chunk.data);
      console.log("\n\n");
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in la\"}]},
       \"stream_mode\": [
         \"values\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^event:/ {
         if (data_content != "") {
             print data_content "\n"
         }
         sub(/^event: /, "Receiving event of type: ", $0)
         printf "%s...\n", $0
         data_content = ""
     }
     /^data:/ {
         sub(/^data: /, "", $0)
         data_content = $0
     }
     END {
         if (data_content != "") {
             print data_content "\n"
         }
     }
     ' 
    ```


Output:

    Receiving new event of type: metadata...
    {"run_id": "f08791ce-0a3d-44e0-836c-ff62cd2e2786"}



    Receiving new event of type: values...
    {
      "messages": [
        {
          "role": "human",
          "content": "what's the weather in la"
        }
      ]
    }



    Receiving new event of type: values...
    {
      "messages": [
        {
          "content": "what's the weather in la",
          "type": "human",
          ...
        },
        {
          "content": "",
          "type": "ai",
          "tool_calls": [
            {
              "name": "tavily_search_results_json",
              "args": {
                "query": "weather in los angeles"
              },
              "id": "toolu_01E5mSaZWm5rWJnCqmt63v4g"
            }
          ],
          ...
        }
      ]
    }

    ...

    Receiving new event of type: values...
    {
      "messages": [
        {
          "content": "what's the weather in la",
          "type": "human",
          ...
        },
        {
          "content": "",
          "type": "ai",
          "tool_calls": [
            {
              "name": "tavily_search_results_json",
              "args": {
                "query": "weather in los angeles"
              },
              "id": "toolu_01E5mSaZWm5rWJnCqmt63v4g"
            }
          ],
          ...
        }
        {
          "content": [
            {
              "url": "https://www.weatherapi.com/",
              "content": "{\"location\": {\"name\": \"Los Angeles\", \"region\": \"California\", \"country\": \"United States of America\", \"lat\": 34.05, \"lon\": -118.24, \"tz_id\": \"America/Los_Angeles\", \"localtime_epoch\": 1716310320, \"localtime\": \"2024-05-21 9:52\"}, \"current\": {\"last_updated_epoch\": 1716309900, \"last_updated\": \"2024-05-21 09:45\", \"temp_c\": 16.7, \"temp_f\": 62.1, \"is_day\": 1, \"condition\": {\"text\": \"Overcast\", \"icon\": \"//cdn.weatherapi.com/weather/64x64/day/122.png\", \"code\": 1009}, \"wind_mph\": 8.1, \"wind_kph\": 13.0, \"wind_degree\": 250, \"wind_dir\": \"WSW\", \"pressure_mb\": 1015.0, \"pressure_in\": 29.97, \"precip_mm\": 0.0, \"precip_in\": 0.0, \"humidity\": 65, \"cloud\": 100, \"feelslike_c\": 16.7, \"feelslike_f\": 62.1, \"vis_km\": 16.0, \"vis_miles\": 9.0, \"uv\": 5.0, \"gust_mph\": 12.5, \"gust_kph\": 20.2}}"
            }
          ],
          "type": "tool",
          "name": "tavily_search_results_json",
          "tool_call_id": "toolu_01E5mSaZWm5rWJnCqmt63v4g"
          ...
        },
        {
          "content": "Based on the weather API results, the current weather in Los Angeles is overcast with a temperature of around 62┬░F (17┬░C). There are light winds from the west-southwest around 8-13 mph. The humidity is 65% and visibility is good at 9 miles. Overall, mild spring weather conditions in LA.",
          "type": "ai",
          ...
        }
      ]
    }



    Receiving new event of type: end...
    None
    
    
    


If we want to just get the final result, we can use this endpoint and just keep track of the last value we received


=== "Python"

    ```python
    final_answer = None
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id,
        input=input,
        stream_mode="values"
    ):
        if chunk.event == "values":
            final_answer = chunk.data
    ```

=== "Javascript"

    ```js
    let finalAnswer;
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input,
        streamMode: "values"
      }
    );
    for await (const chunk of streamResponse) {
      finalAnswer = chunk.data;
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
     --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
     --header 'Content-Type: application/json' \
     --data "{
       \"assistant_id\": \"agent\",
       \"input\": {\"messages\": [{\"role\": \"human\", \"content\": \"what's the weather in la\"}]},
       \"stream_mode\": [
         \"values\"
       ]
     }" | \
     sed 's/\r$//' | \
     awk '
     /^data:/ { 
         sub(/^data: /, "", $0)   
         data_content = $0          
     }    
     END {                                               
         if (data_content != "") {
             print data_content
         }
     }         
     '
    ```


Output:

    {
      "messages": [
        {
          "content": "what's the weather in la",
          "type": "human",
          ...
        },
        {
          "type": "ai",
          "tool_calls": [
            {
              "name": "tavily_search_results_json",
              "args": {
                "query": "weather in los angeles"
              },
              "id": "toolu_01E5mSaZWm5rWJnCqmt63v4g"
            }
          ],
          ...
        }
        {
          "content": [
            {
              "url": "https://www.weatherapi.com/",
              "content": "{\"location\": {\"name\": \"Los Angeles\", \"region\": \"California\", \"country\": \"United States of America\", \"lat\": 34.05, \"lon\": -118.24, \"tz_id\": \"America/Los_Angeles\", \"localtime_epoch\": 1716310320, \"localtime\": \"2024-05-21 9:52\"}, \"current\": {\"last_updated_epoch\": 1716309900, \"last_updated\": \"2024-05-21 09:45\", \"temp_c\": 16.7, \"temp_f\": 62.1, \"is_day\": 1, \"condition\": {\"text\": \"Overcast\", \"icon\": \"//cdn.weatherapi.com/weather/64x64/day/122.png\", \"code\": 1009}, \"wind_mph\": 8.1, \"wind_kph\": 13.0, \"wind_degree\": 250, \"wind_dir\": \"WSW\", \"pressure_mb\": 1015.0, \"pressure_in\": 29.97, \"precip_mm\": 0.0, \"precip_in\": 0.0, \"humidity\": 65, \"cloud\": 100, \"feelslike_c\": 16.7, \"feelslike_f\": 62.1, \"vis_km\": 16.0, \"vis_miles\": 9.0, \"uv\": 5.0, \"gust_mph\": 12.5, \"gust_kph\": 20.2}}"
            }
          ],
          "type": "tool",
          "name": "tavily_search_results_json",
          "tool_call_id": "toolu_01E5mSaZWm5rWJnCqmt63v4g"
          ...
        },
        {
          "content": "Based on the weather API results, the current weather in Los Angeles is overcast with a temperature of around 62┬░F (17┬░C). There are light winds from the west-southwest around 8-13 mph. The humidity is 65% and visibility is good at 9 miles. Overall, mild spring weather conditions in LA.",
          "type": "ai",
          ...
        }
      ]
    }

<a name="docscloudhow-tostest_deploymentmd"></a>

## Test Cloud Deployment

The LangGraph Studio UI connects directly to LangGraph Cloud deployments.

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `LangGraph Cloud`. The `LangGraph Cloud` view contains a list of existing LangGraph Cloud deployments.
1. Select an existing deployment to test with LangGraph Studio.
1. In the top-right corner, select `Open LangGraph Studio`.
1. [Invoke an assistant](#docscloudhow-tosinvoke_studiomd) or [view an existing thread](#docscloudhow-tosthreads_studiomd).

The following video shows these exact steps being carried out:

<video controls allowfullscreen="true" poster="../img/studio_usage_poster.png">
    <source src="../img/studio_usage.mp4" type="video/mp4">
</video>


<a name="docscloudhow-tostest_local_deploymentmd"></a>

## LangGraph Studio With Local Deployment

!!! warning "Browser Compatibility"
    Viewing the studio page of a local LangGraph deployment does not work in Safari. Use Chrome instead.

### Setup

Make sure you have setup your app correctly, by creating a compiled graph, a `.env` file with any environment variables, and a `langgraph.json` config file that points to your environment file and compiled graph. See [here](https://langchain-ai.github.io/langgraph/cloud/deployment/setup/) for more detailed instructions.

After you have your app setup, head into the directory with your `langgraph.json` file and call `langgraph dev` to start the API server in watch mode which means it will restart on code changes, which is ideal for local testing. If the API server start correctly you should see logs that look something like this:

>    Ready!
> 
>    - API: [http://localhost:2024](http://localhost:2024/)
>     
>    - Docs: http://localhost:2024/docs
>     
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

Read this [reference](https://langchain-ai.github.io/langgraph/cloud/reference/cli/#up) to learn about all the options for starting the API server.

### Access Studio

Once you have successfully started the API server, you can access the studio by going to the following URL: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024` (see warning above if using Safari).

If everything is working correctly you should see the studio show up looking something like this (with your graph diagram on the left hand side):

![LangGraph Studio](./img/studio_screenshot.png)

### Use the Studio for Testing

To learn about how to use the studio for testing, read the [LangGraph Studio how-tos](https://langchain-ai.github.io/langgraph/cloud/how-tos/#langgraph-studio).

<a name="docscloudhow-tosthreads_studiomd"></a>

## Interacting with Threads in Studio

### View Thread

1. In the top of the right-hand pane, select the `New Thread` dropdown menu to view existing threads.
1. View the state of the thread (i.e. the output) in the right-hand pane.
1. To create a new thread, select `+ New Thread`.

The following video shows these exact steps being carried out:

<video controls="true" allowfullscreen="true" poster="../img/studio_threads_poster.png">
    <source src="../img/studio_threads.mp4" type="video/mp4">
</video>

### Edit Thread State

The LangGraph Studio UI contains features for editing thread state. Explore these features in the right-hand pane. Select the `Edit` icon, modify the desired state, and then select `Fork` to invoke the assistant with the updated state.

The following video shows how to edit a thread in the studio:

<video controls allowfullscreen="true" poster="../img/studio_forks_poster.png">
    <source src="../img/studio_forks.mp4" type="video/mp4">
</video>


<a name="docscloudhow-toswebhooksmd"></a>

## Use Webhooks

You may wish to use webhooks in your client, especially when using async streams in case you want to update something in your service once the API call to LangGraph Cloud has finished running. To do so, you will need to expose an endpoint that can accept POST requests, and then pass it to your API request in the "webhook" parameter.

Currently, the SDK has not exposed this endpoint but you can access it through curl commands as follows.

The following endpoints accept `webhook` as a parameter: 

- Create Run -> POST /thread/{thread_id}/runs
- Create Thread Cron -> POST /thread/{thread_id}/runs/crons
- Stream Run -> POST /thread/{thread_id}/runs/stream
- Wait Run -> POST /thread/{thread_id}/runs/wait
- Create Cron -> POST /runs/crons
- Stream Run Stateless -> POST /runs/stream
- Wait Run Stateless -> POST /runs/wait

In this example, we will show calling a webhook after streaming a run. 

### Setup

First, let's setup our assistant and thread:

=== "Python"

    ```python
    from langgraph_sdk import get_client

    client = get_client(url=<DEPLOYMENT_URL>)
    # Using the graph deployed with the name "agent"
    assistant_id = "agent"
    # create thread
    thread = await client.threads.create()
    print(thread)
    ```

=== "Javascript"

    ```js
    import { Client } from "@langchain/langgraph-sdk";

    const client = new Client({ apiUrl: <DEPLOYMENT_URL> });
    // Using the graph deployed with the name "agent"
    const assistantID = "agent";
    // create thread
    const thread = await client.threads.create();
    console.log(thread);
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/assistants/search \
        --header 'Content-Type: application/json' \
        --data '{
            "limit": 10,
            "offset": 0
        }' | jq -c 'map(select(.config == null or .config == {})) | .[0]' && \
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads \
        --header 'Content-Type: application/json' \
        --data '{}'
    ```

Output:

    {
        'thread_id': '9dde5490-2b67-47c8-aa14-4bfec88af217', 
        'created_at': '2024-08-30T23:07:38.242730+00:00', 
        'updated_at': '2024-08-30T23:07:38.242730+00:00', 
        'metadata': {}, 
        'status': 'idle', 
        'config': {}, 
        'values': None
    }

### Use graph with a webhook

To invoke a run with a webhook, we specify the `webhook` parameter with the desired endpoint when creating a run. Webhook requests are triggered by the end of a run.

For example, if we can receive requests at `https://my-server.app/my-webhook-endpoint`, we can pass this to `stream`:

=== "Python"

    ```python
    # create input
    input = { "messages": [{ "role": "user", "content": "Hello!" }] }

    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input=input,
        stream_mode="events",
        webhook="https://my-server.app/my-webhook-endpoint"
    ):
        # Do something with the stream output
        pass
    ```

=== "Javascript"

    ```js
    // create input
    const input = { messages: [{ role: "human", content: "Hello!" }] };

    // stream events
    const streamResponse = client.runs.stream(
      thread["thread_id"],
      assistantID,
      {
        input: input,
        webhook: "https://my-server.app/my-webhook-endpoint"
      }
    );
    for await (const chunk of streamResponse) {
      // Do something with the stream output
    }
    ```

=== "CURL"

    ```bash
    curl --request POST \
        --url <DEPLOYMENT_URL>/threads/<THREAD_ID>/runs/stream \
        --header 'Content-Type: application/json' \
        --data '{
            "assistant_id": <ASSISTANT_ID>,
            "input" : {"messages":[{"role": "user", "content": "Hello!"}]},
            "webhook": "https://my-server.app/my-webhook-endpoint"
        }'
    ```

The schema for the payload sent to `my-webhook-endpoint` is that of a [run](#runs). See [API Reference](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html#model/run) for more detail. Note that the run input, configuration, etc. are included in the `kwargs` field.

#### Signing webhook requests

To sign the webhook requests, we can specify a token parameter in the webhook URL, e.g.,
```
https://my-server.app/my-webhook-endpoint?token=...
```

The server should then extract the token from the request's parameters and validate it before processing the payload.


<a name="docscloudquick_startmd"></a>

## Quickstart: Deploy on LangGraph Cloud

!!! note "Prerequisites"

    Before you begin, ensure you have the following:

    - [GitHub account](https://github.com/)
    - [LangSmith account](https://smith.langchain.com/)

### Create a repository on GitHub

To deploy a LangGraph application to **LangGraph Cloud**, your application code must reside in a GitHub repository. Both public and private repositories are supported.

You can deploy any [LangGraph Application](#docsconceptsapplication_structuremd) to LangGraph Cloud.

For this guide, we'll use the pre-built Python [**ReAct Agent**](https://github.com/langchain-ai/react-agent) template.

??? note "Get Required API Keys for the ReAct Agent template"

    This **ReAct Agent** application requires an API key from [Anthropic](https://console.anthropic.com/) and [Tavily](https://app.tavily.com/). You can get these API keys by signing up on their respective websites.

    **Alternative**: If you'd prefer a scaffold application that doesn't require API keys, use the [**New LangGraph Project**](https://github.com/langchain-ai/new-langgraph-project) template instead of the **ReAct Agent** template.


1. Go to the [ReAct Agent](https://github.com/langchain-ai/react-agent) repository.
2. Fork the repository to your GitHub account by clicking the `Fork` button in the top right corner.

### Deploy to LangGraph Cloud

??? note "1. Log in to [LangSmith](https://smith.langchain.com/)"

    <figure markdown="1">
    [![Login to LangSmith](deployment/img/01_login.png){: style="max-height:300px"}](deployment/img/01_login.png)
    <figcaption>
    Go to [LangSmith](https://smith.langchain.com/) and log in. If you don't have an account, you can sign up for free.
    </figcaption>
    </figure>


??? note "2. Click on <em>LangGraph Platform</em> (the left sidebar)"

    <figure markdown="1">
    [![Login to LangSmith](deployment/img/02_langgraph_platform.png){: style="max-height:300px"}](deployment/img/02_langgraph_platform.png)
    <figcaption>
    Select **LangGraph Platform** from the left sidebar.
    </figcaption>
    </figure>

??? note "3. Click on + New Deployment (top right corner)"

    <figure markdown="1">
    [![Login to LangSmith](deployment/img/03_deployments_page.png){: style="max-height:300px"}](deployment/img/03_deployments_page.png)
    <figcaption>
    Click on **+ New Deployment** to create a new deployment. This button is located in the top right corner.
    It'll open a new modal where you can fill out the required fields.
    </figcaption>
    </figure>

??? note "4. Click on Import from GitHub (first time users)"

    <figure markdown="1">
    [![image](deployment/img/04_create_new_deployment.png)](deployment/img/04_create_new_deployment.png)
    <figcaption>
    Click on **Import from GitHub** and follow the instructions to connect your GitHub account. This step is needed for **first-time users** or to add private repositories that haven't been connected before.</figcaption>
        </figure>

??? note "5. Select the repository, configure ENV vars etc"

    <figure markdown="1">
    [![image](deployment/img/05_configure_deployment.png){: style="max-height:300px"}](deployment/img/05_configure_deployment.png)
    <figcaption>
    Select the <strong>repository</strong>, add env variables and secrets, and set other configuration options.
    </figcaption>
    </figure>

    - **Repository**: Select the repository you forked earlier (or any other repository you want to deploy).
    - Set the secrets and environment variables required by your application. For the **ReAct Agent** template, you need to set the following secrets:
        - **ANTHROPIC_API_KEY**: Get an API key from [Anthropic](https://console.anthropic.com/).
        - **TAVILY_API_KEY**: Get an API key on the [Tavily website](https://app.tavily.com/).

??? note "6. Click Submit to Deploy!"


    <figure markdown="1">
    [![image](deployment/img/05_configure_deployment.png){: style="max-height:300px"}](deployment/img/05_configure_deployment.png)
    <figcaption>
        Please note that this step may ~15 minutes to complete. You can check the status of your deployment in the **Deployments** view.
        Click the <strong>Submit</strong> button at the top right corner to deploy your application.
    </figcaption>
    </figure>


### Lagraph Studio Web UI

Once your application is deployed, you can test it in **LangGraph Studio**. 

??? note "1. Click on an existing deployment"

    <figure markdown="1">
    [![image](deployment/img/07_deployments_page.png){: style="max-height:300px"}](deployment/img/07_deployments_page.png)
    <figcaption>
        Click on the deployment you just created to view more details.
    </figcaption>
    </figure>

??? note "2. Click on LangGraph Studio"

    <figure markdown="1">
    [![image](deployment/img/08_deployment_view.png){: style="max-height:300px"}](deployment/img/08_deployment_view.png)
    <figcaption>
        Click on the <strong>LangGraph Studio</strong> button to open LangGraph Studio.
    </figcaption>
    </figure>

<figure markdown="1">
[![image](deployment/img/09_langgraph_studio.png){: style="max-height:400px"}](deployment/img/09_langgraph_studio.png)
<figcaption>
    Sample graph run in LangGraph Studio.
</figcaption>
</figure>

### Test the API

!!! note

    The API calls below are for the **ReAct Agent** template. If you're deploying a different application, you may need to adjust the API calls accordingly.

Before using, you need to get the `URL` of your LangGraph deployment. You can find this in the `Deployment` view. Click the `URL` to copy it to the clipboard.

You also need to make sure you have set up your API key properly, so you can authenticate with LangGraph Cloud.

```shell
export LANGSMITH_API_KEY=...
```

=== "Python SDK (Async)"

    **Install the LangGraph Python SDK**

    ```shell
    pip install langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```python
    from langgraph_sdk import get_client

    client = get_client(url="your-deployment-url", api_key="your-langsmith-api-key")

    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Python SDK (Sync)"

    **Install the LangGraph Python SDK**

    ```shell
    pip install langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```python
    from langgraph_sdk import get_sync_client

    client = get_sync_client(url="your-deployment-url", api_key="your-langsmith-api-key")

    for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript SDK"

    **Install the LangGraph JS SDK**

    ```shell
    npm install @langchain/langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```js
    const { Client } = await import("@langchain/langgraph-sdk");

    const client = new Client({ apiUrl: "your-deployment-url", apiKey: "your-langsmith-api-key" });

    const streamResponse = client.runs.stream(
        null, // Threadless run
        "agent", // Assistant ID
        {
            input: {
                "messages": [
                    { "role": "user", "content": "What is LangGraph?"}
                ]
            },
            streamMode: "messages",
        }
    );

    for await (const chunk of streamResponse) {
        console.log(`Receiving new event of type: ${chunk.event}...`);
        console.log(JSON.stringify(chunk.data));
        console.log("\n\n");
    }
    ```

=== "Rest API"

    ```bash
    curl -s --request POST \
        --url <DEPLOYMENT_URL> \
        --header 'Content-Type: application/json' \
        --data "{
            \"assistant_id\": \"agent\",
            \"input\": {
                \"messages\": [
                    {
                        \"role\": \"human\",
                        \"content\": \"What is LangGraph?\"
                    }
                ]
            },
            \"stream_mode\": \"updates\"
        }" 
    ```


### Next Steps

Congratulations! If you've worked your way through this tutorial you are well on your way to becoming a LangGraph Cloud expert. Here are some other resources to check out to help you out on the path to expertise:

#### LangGraph Framework

- **[LangGraph Tutorial](../tutorials/introduction.ipynb)**: Get started with LangGraph framework.
- **[LangGraph Concepts](#docsconceptsindexmd)**: Learn the foundational concepts of LangGraph.
- **[LangGraph How-to Guides](#docshow-tosindexmd)**: Guides for common tasks with LangGraph.
 
#### ­ƒôÜ Learn More about LangGraph Platform

Expand your knowledge with these resources:

- **[LangGraph Platform Concepts](#langgraph-platform)**: Understand the foundational concepts of the LangGraph Platform.
- **[LangGraph Platform How-to Guides](#langgraph-platform)**: Discover step-by-step guides to build and deploy applications.
- **[Launch Local LangGraph Server](#docstutorialslanggraph-platformlocal-servermd)**: This quick start guide shows how to start a LangGraph Server locally for the **ReAct Agent** template. The steps are similar for other templates.




<a name="docscloudreferenceapiapi_refmd"></a>

## API Reference

The LangGraph Cloud API reference is available with each deployment at the `/docs` URL path (e.g. `http://localhost:8124/docs`).

Click <a href="/langgraph/cloud/reference/api/api_ref.html" target="_blank">here</a> to view the API reference.

### Authentication

For deployments to LangGraph Cloud, authentication is required. Pass the `X-Api-Key` header with each request to the LangGraph Cloud API. The value of the header should be set to a valid LangSmith API key for the organization where the API is deployed.

Example `curl` command:
```shell
curl --request POST \
  --url http://localhost:8124/assistants/search \
  --header 'Content-Type: application/json' \
  --header 'X-Api-Key: LANGSMITH_API_KEY' \
  --data '{
  "metadata": {},
  "limit": 10,
  "offset": 0
}'  
```


<a name="docscloudreferenceclimd"></a>

## LangGraph CLI

The LangGraph command line interface includes commands to build and run a LangGraph Cloud API server locally in [Docker](https://www.docker.com/). For development and testing, you can use the CLI to deploy a local API server as an alternative to the [Studio desktop app](#docsconceptslanggraph_studiomd).

### Installation

1. Ensure that Docker is installed (e.g. `docker --version`).
2. Install the `langgraph-cli` package:
 
    === "pip"
        ```bash    
        pip install langgraph-cli
        ```

    === "Homebrew (MacOS only)"
        ```bash
        brew install langgraph-cli
        ```
 
3. Run the command `langgraph --help` to confirm that the CLI is installed.

[](){#langgraph.json}

### Configuration File {#configuration-file}

The LangGraph CLI requires a JSON configuration file with the following keys:

| Key                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dependencies`     | **Required**. Array of dependencies for LangGraph Cloud API server. Dependencies can be one of the following: (1) `"."`, which will look for local Python packages, (2) `pyproject.toml`, `setup.py` or `requirements.txt` in the app directory `"./local_package"`, or (3) a package name.                                                                                                                                                                                                                                                  |
| `graphs`           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./your_package/your_file.py:variable`, where `variable` is an instance of `langgraph.graph.state.CompiledStateGraph`</li><li>`./your_package/your_file.py:make_graph`, where `make_graph` is a function that takes a config dictionary (`langchain_core.runnables.RunnableConfig`) and creates an instance of `langgraph.graph.state.StateGraph` / `langgraph.graph.state.CompiledStateGraph`.</li></ul> |
| `auth`             | _(Added in v0.0.11)_ Auth configuration containing the path to your authentication handler. Example: `./your_package/auth.py:auth`, where `auth` is an instance of `langgraph_sdk.Auth`. See [authentication guide](#docsconceptsauthmd) for details.                                                                                                                                                                                                                                                                                                                      |
| `env`              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `store`            | Configuration for adding semantic search to the BaseStore. Contains the following fields: <ul><li>`index`: Configuration for semantic search indexing with fields:<ul><li>`embed`: Embedding provider (e.g., "openai:text-embedding-3-small") or path to custom embedding function</li><li>`dims`: Dimension size of the embedding model. Used to initialize the vector table.</li><li>`fields` (optional): List of fields to index. Defaults to `["$"]`, meaningto index entire documents. Can be specific fields like `["text", "summary", "some.value"]`</li></ul></li></ul>                                                                  |
| `python_version`   | `3.11` or `3.12`. Defaults to `3.11`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `pip_config_file`  | Path to `pip` config file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `dockerfile_lines` | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

<div class="admonition tip">
    <p class="admonition-title">Note</p>
    <p>
        The LangGraph CLI defaults to using the configuration file <strong>langgraph.json</strong> in the current directory.
    </p>
</div>

#### Examples

##### Basic Configuration

```json
{
  "dependencies": ["."],
  "graphs": {
    "chat": "./chat/graph.py:graph"
  }
}
```

##### Adding semantic search to the store

All deployments come with a DB-backed BaseStore. Adding an "index" configuration to your `langgraph.json` will enable [semantic search](#docsclouddeploymentsemantic_searchmd) within the BaseStore of your deployment.

The `fields` configuration determines which parts of your documents to embed:
- If omitted or set to `["$"]`, the entire document will be embedded
- To embed specific fields, use JSON path notation: `["metadata.title", "content.text"]`
- Documents missing specified fields will still be stored but won't have embeddings for those fields
- You can still override which fields to embed on a specific item at `put` time using the `index` parameter

```json
{
  "dependencies": ["."],
  "graphs": {
    "memory_agent": "./agent/graph.py:graph"
  },
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["$"]
    }
  }
}
```

!!! note "Common model dimensions"
        - openai:text-embedding-3-large: 3072
        - openai:text-embedding-3-small: 1536
        - openai:text-embedding-ada-002: 1536
        - cohere:embed-english-v3.0: 1024
        - cohere:embed-english-light-v3.0: 384
        - cohere:embed-multilingual-v3.0: 1024
        - cohere:embed-multilingual-light-v3.0: 384

##### Semantic search with a custom embedding function

If you want to use semantic search with a custom embedding function, you can pass a path to a custom embedding function:

```json
{
  "dependencies": ["."],
  "graphs": {
    "memory_agent": "./agent/graph.py:graph"
  },
  "store": {
    "index": {
      "embed": "./embeddings.py:embed_texts",
      "dims": 768,
      "fields": ["text", "summary"]
    }
  }
}
```

The `embed` field in store configuration can reference a custom function that takes a list of strings and returns a list of embeddings. Example implementation:

```python
## embeddings.py
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Custom embedding function for semantic search."""
    # Implementation using your preferred embedding model
    return [[0.1, 0.2, ...] for _ in texts]  # dims-dimensional vectors
```

##### Adding custom authentication

```json
{
  "dependencies": ["."],
  "graphs": {
    "chat": "./chat/graph.py:graph"
  },
  "auth": {
    "path": "./auth.py:auth",
    "openapi": {
      "securitySchemes": {
        "apiKeyAuth": {
          "type": "apiKey",
          "in": "header",
          "name": "X-API-Key"
        }
      },
      "security": [
        {"apiKeyAuth": []}
      ]
    },
    "disable_studio_auth": false
  }
}
```

See the [authentication conceptual guide](#docsconceptsauthmd) for details, and the [setting up custom authentication](#docstutorialsauthgetting_startedmd) guide for a practical walk through of the process.

### Commands

The base command for the LangGraph CLI is `langgraph`.

**Usage**

```
langgraph [OPTIONS] COMMAND [ARGS]
```

#### `dev`

Run LangGraph API server in development mode with hot reloading and debugging capabilities. This lightweight server requires no Docker installation and is suitable for development and testing. State is persisted to a local directory.

!!! note "Python only"

    Currently, the CLI only supports Python >= 3.11.
    JS support is coming soon.

**Installation**

This command requires the "inmem" extra to be installed:

```bash
pip install -U "langgraph-cli[inmem]"
```

**Usage**

```
langgraph dev [OPTIONS]
```

**Options**

| Option                        | Default          | Description                                                                         |
| ----------------------------- | ---------------- | ----------------------------------------------------------------------------------- |
| `-c, --config FILE`           | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables |
| `--host TEXT`                 | `127.0.0.1`      | Host to bind the server to                                                          |
| `--port INTEGER`              | `2024`           | Port to bind the server to                                                          |
| `--no-reload`                 |                  | Disable auto-reload                                                                 |
| `--n-jobs-per-worker INTEGER` |                  | Number of jobs per worker. Default is 10                                            |
| `--no-browser`                |                  | Disable automatic browser opening                                                   |
| `--debug-port INTEGER`        |                  | Port for debugger to listen on                                                      |
| `--help`                      |                  | Display command documentation                                                       |

#### `build`

Build LangGraph Cloud API server Docker image.

**Usage**

```
langgraph build [OPTIONS]
```

**Options**

| Option               | Default          | Description                                                                                                                  |
| -------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `--platform TEXT`    |                  | Target platform(s) to build the Docker image for. Example: `langgraph build --platform linux/amd64,linux/arm64`              |
| `-t, --tag TEXT`     |                  | **Required**. Tag for the Docker image. Example: `langgraph build -t my-image`                                               |
| `--pull / --no-pull` | `--pull`         | Build with latest remote Docker image. Use `--no-pull` for running the LangGraph Cloud API server with locally built images. |
| `-c, --config FILE`  | `langgraph.json` | Path to configuration file declaring dependencies, graphs and environment variables.                                         |
| `--help`             |                  | Display command documentation.                                                                                               |

#### `up`

Start LangGraph API server. For local testing, requires a LangSmith API key with access to LangGraph Cloud closed beta. Requires a license key for production use.

**Usage**

```
langgraph up [OPTIONS]
```

**Options**

| Option                       | Default                   | Description                                                                                                             |
| ---------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `--wait`                     |                           | Wait for services to start before returning. Implies --detach                                                           |
| `--postgres-uri TEXT`        | Local database            | Postgres URI to use for the database.                                                                                   |
| `--watch`                    |                           | Restart on file changes                                                                                                 |
| `--debugger-base-url TEXT`   | `http://127.0.0.1:[PORT]` | URL used by the debugger to access LangGraph API.                                                                       |
| `--debugger-port INTEGER`    |                           | Pull the debugger image locally and serve the UI on specified port                                                      |
| `--verbose`                  |                           | Show more output from the server logs.                                                                                  |
| `-c, --config FILE`          | `langgraph.json`          | Path to configuration file declaring dependencies, graphs and environment variables.                                    |
| `-d, --docker-compose FILE`  |                           | Path to docker-compose.yml file with additional services to launch.                                                     |
| `-p, --port INTEGER`         | `8123`                    | Port to expose. Example: `langgraph up --port 8000`                                                                     |
| `--pull / --no-pull`         | `pull`                    | Pull latest images. Use `--no-pull` for running the server with locally-built images. Example: `langgraph up --no-pull` |
| `--recreate / --no-recreate` | `no-recreate`             | Recreate containers even if their configuration and image haven't changed                                               |
| `--help`                     |                           | Display command documentation.                                                                                          |

#### `dockerfile`

Generate a Dockerfile for building a LangGraph Cloud API server Docker image.

**Usage**

```
langgraph dockerfile [OPTIONS] SAVE_PATH
```

**Options**

| Option              | Default          | Description                                                                                                     |
| ------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- |
| `-c, --config FILE` | `langgraph.json` | Path to the [configuration file](#configuration-file) declaring dependencies, graphs and environment variables. |
| `--help`            |                  | Show this message and exit.                                                                                     |

Example:

```bash
langgraph dockerfile -c langgraph.json Dockerfile
```

This generates a Dockerfile that looks similar to:

```dockerfile
FROM langchain/langgraph-api:3.11

ADD ./pipconf.txt /pipconfig.txt

RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt langchain_community langchain_anthropic langchain_openai wikipedia scikit-learn

ADD ./graphs /deps/__outer_graphs/src
RUN set -ex && \
    for line in '[project]' \
                'name = "graphs"' \
                'version = "0.1"' \
                '[tool.setuptools.package-data]' \
                '"*" = ["**/*"]'; do \
        echo "$line" >> /deps/__outer_graphs/pyproject.toml; \
    done

RUN PIP_CONFIG_FILE=/pipconfig.txt PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"agent": "/deps/__outer_graphs/src/agent.py:graph", "storm": "/deps/__outer_graphs/src/storm.py:graph"}'
```

<a name="docscloudreferenceenv_varmd"></a>

## Environment Variables

The LangGraph Cloud API supports specific environment variables for configuring a deployment.

### `LANGCHAIN_TRACING_SAMPLING_RATE`

Sampling rate for traces sent to LangSmith. Valid values: Any float between `0` and `1`.

See <a href="https://docs.smith.langchain.com/how_to_guides/tracing/sample_traces" target="_blank">LangSmith documentation</a> for more details.

### `LANGGRAPH_AUTH_TYPE`

Type of authentication for the LangGraph Cloud API deployment. Valid values: `langsmith`, `noop`.

For deployments to LangGraph Cloud, this environment variable is set automatically. For local development or deployments where authentication is handled externally (e.g. self-hosted), set this environment variable to `noop`.

### `N_JOBS_PER_WORKER`

Number of jobs per worker for the LangGraph Cloud task queue. Defaults to `10`.


<a name="docscloudreferencesdkpython_sdk_refmd"></a>

## Python SDK Reference

::: langgraph_sdk.client
    handler: python


::: langgraph_sdk.schema
    handler: python

::: langgraph_sdk.auth
    handler: python

::: langgraph_sdk.auth.types
    handler: python

::: langgraph_sdk.auth.exceptions
    handler: python

<a name="docsconceptsagentic_conceptsmd"></a>

## Agent architectures

Many LLM applications implement a particular control flow of steps before and / or after LLM calls. As an example, [RAG](https://github.com/langchain-ai/rag-from-scratch) performs retrieval of documents relevant to a user question, and passes those documents to an LLM in order to ground the model's response in the provided document context. 

Instead of hard-coding a fixed control flow, we sometimes want LLM systems that can pick their own control flow to solve more complex problems! This is one definition of an [agent](https://blog.langchain.dev/what-is-an-agent/): *an agent is a system that uses an LLM to decide the control flow of an application.* There are many ways that an LLM can control application:

- An LLM can route between two potential paths
- An LLM can decide which of many tools to call
- An LLM can decide whether the generated answer is sufficient or more work is needed

As a result, there are many different types of [agent architectures](https://blog.langchain.dev/what-is-a-cognitive-architecture/), which give an LLM varying levels of control. 

![Agent Types](img/agent_types.png)

### Router

A router allows an LLM to select a single step from a specified set of options. This is an agent architecture that exhibits a relatively limited level of control because the LLM usually focuses on making a single decision and produces a specific output from limited set of pre-defined options. Routers typically employ a few different concepts to achieve this.

#### Structured Output

Structured outputs with LLMs work by providing a specific format or schema that the LLM should follow in its response. This is similar to tool calling, but more general. While tool calling typically involves selecting and using predefined functions, structured outputs can be used for any type of formatted response. Common methods to achieve structured outputs include:

1. Prompt engineering: Instructing the LLM to respond in a specific format via the system prompt.
2. Output parsers: Using post-processing to extract structured data from LLM responses.
3. Tool calling: Leveraging built-in tool calling capabilities of some LLMs to generate structured outputs.

Structured outputs are crucial for routing as they ensure the LLM's decision can be reliably interpreted and acted upon by the system. Learn more about [structured outputs in this how-to guide](https://python.langchain.com/docs/how_to/structured_output/).

### Tool calling agent

While a router allows an LLM to make a single decision, more complex agent architectures expand the LLM's control in two key ways:

1. Multi-step decision making: The LLM can make a series of decisions, one after another, instead of just one.
2. Tool access: The LLM can choose from and use a variety of tools to accomplish tasks.

[ReAct](https://arxiv.org/abs/2210.03629) is a popular general purpose agent architecture that combines these expansions, integrating three core concepts. 

1. `Tool calling`: Allowing the LLM to select and use various tools as needed.
2. `Memory`: Enabling the agent to retain and use information from previous steps.
3. `Planning`: Empowering the LLM to create and follow multi-step plans to achieve goals.

This architecture allows for more complex and flexible agent behaviors, going beyond simple routing to enable dynamic problem-solving with multiple steps. You can use it with [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent].

#### Tool calling

Tools are useful whenever you want an agent to interact with external systems. External systems (e.g., APIs) often require a particular input schema or payload, rather than natural language. When we bind an API, for example, as a tool, we give the model awareness of the required input schema. The model will choose to call a tool based upon the natural language input from the user and it will return an output that adheres to the tool's required schema. 

[Many LLM providers support tool calling](https://python.langchain.com/docs/integrations/chat/) and [tool calling interface](https://blog.langchain.dev/improving-core-tool-interfaces-and-docs-in-langchain/) in LangChain is simple: you can simply pass any Python `function` into `ChatModel.bind_tools(function)`.

![Tools](img/tool_call.png)

#### Memory

Memory is crucial for agents, enabling them to retain and utilize information across multiple steps of problem-solving. It operates on different scales:

1. Short-term memory: Allows the agent to access information acquired during earlier steps in a sequence.
2. Long-term memory: Enables the agent to recall information from previous interactions, such as past messages in a conversation.

LangGraph provides full control over memory implementation:

- [`State`](#state): User-defined schema specifying the exact structure of memory to retain.
- [`Checkpointers`](#docsconceptspersistencemd): Mechanism to store state at every step across different interactions.

This flexible approach allows you to tailor the memory system to your specific agent architecture needs. For a practical guide on adding memory to your graph, see [this tutorial](../how-tos/persistence.ipynb).

Effective memory management enhances an agent's ability to maintain context, learn from past experiences, and make more informed decisions over time.

#### Planning

In the ReAct architecture, an LLM is called repeatedly in a while-loop. At each step the agent decides which tools to call, and what the inputs to those tools should be. Those tools are then executed, and the outputs are fed back into the LLM as observations. The while-loop terminates when the agent decides it has enough information to solve the user request and it is not worth calling any more tools.

#### ReAct implementation 

There are several differences between [this](https://arxiv.org/abs/2210.03629) paper and the pre-built [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] implementation:

- First, we use [tool-calling](#tool-calling) to have LLMs call tools, whereas the paper used prompting + parsing of raw output. This is because tool calling did not exist when the paper was written, but is generally better and more reliable.
- Second, we use messages to prompt the LLM, whereas the paper used string formatting. This is because at the time of writing, LLMs didn't even expose a message-based interface, whereas now that's the only interface they expose.
- Third, the paper required all inputs to the tools to be a single string. This was largely due to LLMs not being super capable at the time, and only really being able to generate a single input. Our implementation allows for using tools that require multiple inputs.
- Fourth, the paper only looks at calling a single tool at the time, largely due to limitations in LLMs performance at the time. Our implementation allows for calling multiple tools at a time.
- Finally, the paper asked the LLM to explicitly generate a "Thought" step before deciding which tools to call. This is the "Reasoning" part of "ReAct". Our implementation does not do this by default, largely because LLMs have gotten much better and that is not as necessary. Of course, if you wish to prompt it do so, you certainly can.

### Custom agent architectures

While routers and tool-calling agents (like ReAct) are common, [customizing agent architectures](https://blog.langchain.dev/why-you-should-outsource-your-agentic-infrastructure-but-own-your-cognitive-architecture/) often leads to better performance for specific tasks. LangGraph offers several powerful features for building tailored agent systems:

#### Human-in-the-loop

Human involvement can significantly enhance agent reliability, especially for sensitive tasks. This can involve:

- Approving specific actions
- Providing feedback to update the agent's state
- Offering guidance in complex decision-making processes

Human-in-the-loop patterns are crucial when full automation isn't feasible or desirable. Learn more in our [human-in-the-loop guide](#docsconceptshuman_in_the_loopmd).

#### Parallelization 

Parallel processing is vital for efficient multi-agent systems and complex tasks. LangGraph supports parallelization through its [Send](#send) API, enabling:

- Concurrent processing of multiple states
- Implementation of map-reduce-like operations
- Efficient handling of independent subtasks

For practical implementation, see our [map-reduce tutorial](../how-tos/map-reduce.ipynb).

#### Subgraphs

[Subgraphs](#subgraphs) are essential for managing complex agent architectures, particularly in [multi-agent systems](#docsconceptsmulti_agentmd). They allow:

- Isolated state management for individual agents
- Hierarchical organization of agent teams
- Controlled communication between agents and the main system

Subgraphs communicate with the parent graph through overlapping keys in the state schema. This enables flexible, modular agent design. For implementation details, refer to our [subgraph how-to guide](../how-tos/subgraph.ipynb).

#### Reflection

Reflection mechanisms can significantly improve agent reliability by:

1. Evaluating task completion and correctness
2. Providing feedback for iterative improvement
3. Enabling self-correction and learning

While often LLM-based, reflection can also use deterministic methods. For instance, in coding tasks, compilation errors can serve as feedback. This approach is demonstrated in [this video using LangGraph for self-corrective code generation](https://www.youtube.com/watch?v=MvNdgmM7uyc).

By leveraging these features, LangGraph enables the creation of sophisticated, task-specific agent architectures that can handle complex workflows, collaborate effectively, and continuously improve their performance.


<a name="docsconceptsapplication_structuremd"></a>

## Application Structure

!!! info "Prerequisites"

    - [LangGraph Server](#docsconceptslanggraph_servermd)
    - [LangGraph Glossary](#docsconceptslow_levelmd)

### Overview

A LangGraph application consists of one or more graphs, a LangGraph API Configuration file (`langgraph.json`), a file that specifies dependencies, and an optional .env file that specifies environment variables.

This guide shows a typical structure for a LangGraph application and shows how the required information to deploy a LangGraph application using the LangGraph Platform is specified.

### Key Concepts

To deploy using the LangGraph Platform, the following information should be provided:

1. A [LangGraph API Configuration file](#configuration-file) (`langgraph.json`) that specifies the dependencies, graphs, environment variables to use for the application.
2. The [graphs](#graphs) that implement the logic of the application.
3. A file that specifies [dependencies](#dependencies) required to run the application.
4. [Environment variable](#environment-variables) that are required for the application to run.

### File Structure

Below are examples of directory structures for Python and JavaScript applications:

=== "Python (requirements.txt)"

    ```plaintext
    my-app/
    Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
    Ôöé   Ôö£ÔöÇÔöÇ utils # utilities for your graph
    Ôöé   Ôöé   Ôö£ÔöÇÔöÇ __init__.py
    Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.py # tools for your graph
    Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.py # node functions for you graph
    Ôöé   Ôöé   ÔööÔöÇÔöÇ state.py # state definition of your graph
    Ôöé   Ôö£ÔöÇÔöÇ requirements.txt # package dependencies
    Ôöé   Ôö£ÔöÇÔöÇ __init__.py
    Ôöé   ÔööÔöÇÔöÇ agent.py # code for constructing your graph
    Ôö£ÔöÇÔöÇ .env # environment variables
    ÔööÔöÇÔöÇ langgraph.json # configuration file for LangGraph
    ```
=== "Python (pyproject.toml)"

    ```plaintext
    my-app/
    Ôö£ÔöÇÔöÇ my_agent # all project code lies within here
    Ôöé   Ôö£ÔöÇÔöÇ utils # utilities for your graph
    Ôöé   Ôöé   Ôö£ÔöÇÔöÇ __init__.py
    Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.py # tools for your graph
    Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.py # node functions for you graph
    Ôöé   Ôöé   ÔööÔöÇÔöÇ state.py # state definition of your graph
    Ôöé   Ôö£ÔöÇÔöÇ __init__.py
    Ôöé   ÔööÔöÇÔöÇ agent.py # code for constructing your graph
    Ôö£ÔöÇÔöÇ .env # environment variables
    Ôö£ÔöÇÔöÇ langgraph.json  # configuration file for LangGraph
    ÔööÔöÇÔöÇ pyproject.toml # dependencies for your project
    ```

=== "JS (package.json)"

    ```plaintext
    my-app/
    Ôö£ÔöÇÔöÇ src # all project code lies within here
    Ôöé   Ôö£ÔöÇÔöÇ utils # optional utilities for your graph
    Ôöé   Ôöé   Ôö£ÔöÇÔöÇ tools.ts # tools for your graph
    Ôöé   Ôöé   Ôö£ÔöÇÔöÇ nodes.ts # node functions for you graph
    Ôöé   Ôöé   ÔööÔöÇÔöÇ state.ts # state definition of your graph
    Ôöé   ÔööÔöÇÔöÇ agent.ts # code for constructing your graph
    Ôö£ÔöÇÔöÇ package.json # package dependencies
    Ôö£ÔöÇÔöÇ .env # environment variables
    ÔööÔöÇÔöÇ langgraph.json # configuration file for LangGraph
    ```

!!! note

    The directory structure of a LangGraph application can vary depending on the programming language and the package manager used.


### Configuration File

The `langgraph.json` file is a JSON file that specifies the dependencies, graphs, environment variables, and other settings required to deploy a LangGraph application.

The file supports specification of the following information:


| Key                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `dependencies`     | **Required**. Array of dependencies for LangGraph API server. Dependencies can be one of the following: (1) `"."`, which will look for local Python packages, (2) `pyproject.toml`, `setup.py` or `requirements.txt` in the app directory `"./local_package"`, or (3) a package name.                                                                                                                                                                                                                                                        |
| `graphs`           | **Required**. Mapping from graph ID to path where the compiled graph or a function that makes a graph is defined. Example: <ul><li>`./your_package/your_file.py:variable`, where `variable` is an instance of `langgraph.graph.state.CompiledStateGraph`</li><li>`./your_package/your_file.py:make_graph`, where `make_graph` is a function that takes a config dictionary (`langchain_core.runnables.RunnableConfig`) and creates an instance of `langgraph.graph.state.StateGraph` / `langgraph.graph.state.CompiledStateGraph`.</li></ul> |
| `env`              | Path to `.env` file or a mapping from environment variable to its value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `python_version`   | `3.11` or `3.12`. Defaults to `3.11`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `pip_config_file`  | Path to `pip` config file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `dockerfile_lines` | Array of additional lines to add to Dockerfile following the import from parent image.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
!!! tip

    The LangGraph CLI defaults to using the configuration file **langgraph.json** in the current directory.


#### Examples

=== "Python"

    * The dependencies involve a custom local package and the `langchain_openai` package.
    * A single graph will be loaded from the file `./your_package/your_file.py` with the variable `variable`.
    * The environment variables are loaded from the `.env` file.

    ```json
    {
        "dependencies": [
            "langchain_openai",
            "./your_package"
        ],
        "graphs": {
            "my_agent": "./your_package/your_file.py:agent"
        },
        "env": "./.env"
    }
    ```

=== "JavaScript"

    * The dependencies will be loaded from a dependency file in the local directory (e.g., `package.json`).
    * A single graph will be loaded from the file `./your_package/your_file.js` with the function `agent`.
    * The environment variable `OPENAI_API_KEY` is set inline.

    ```json
    {
        "dependencies": [
            "."
        ],
        "graphs": {
            "my_agent": "./your_package/your_file.js:agent"
        },
        "env": {
            "OPENAI_API_KEY": "secret-key"
        }
    }
    ```

### Dependencies

A LangGraph application may depend on other Python packages or JavaScript libraries (depending on the programming language in which the application is written).

You will generally need to specify the following information for dependencies to be set up correctly:

1. A file in the directory that specifies the dependencies (e.g., `requirements.txt`, `pyproject.toml`, or `package.json`).
2. A `dependencies` key in the [LangGraph configuration file](#configuration-file) that specifies the dependencies required to run the LangGraph application.
3. Any additional binaries or system libraries can be specified using `dockerfile_lines` key in the [LangGraph configuration file](#configuration-file).

### Graphs

Use the `graphs` key in the [LangGraph configuration file](#configuration-file) to specify which graphs will be available in the deployed LangGraph application.

You can specify one or more graphs in the configuration file. Each graph is identified by a name (which should be unique) and a path for either: (1) the compiled graph or (2) a function that makes a graph is defined.

### Environment Variables

If you're working with a deployed LangGraph application locally, you can configure environment variables in the `env` key of the [LangGraph configuration file](#configuration-file).

For a production deployment, you will typically want to configure the environment variables in the deployment environment.

### Related

Please see the following resources for more information:

- How-to guides for [Application Structure](#application-structure).


<a name="docsconceptsassistantsmd"></a>

## Assistants

!!! info "Prerequisites"

    - [LangGraph Server](#docsconceptslanggraph_servermd)

When building agents, it is fairly common to make rapid changes that *do not* alter the graph logic. For example, simply changing prompts or the LLM selection can have significant impacts on the behavior of the agents. Assistants offer an easy way to make and save these types of changes to agent configuration. This can have at least two use-cases:

* Assistants give developers a quick and easy way to modify and version agents for experimentation.
* Assistants can be modified via LangGraph Studio, offering a no-code way to configure agents  (e.g., for business users). 

Assistants build off the concept of ["configuration"](#configuration). 
While ["configuration"](#configuration) is available in the open source LangGraph library as well,  assistants are only present in [LangGraph Platform](#docsconceptslanggraph_platformmd).
This is because Assistants are tightly coupled to your deployed graph, and so we can only make them available when we are also deploying the graphs.

### Configuring Assistants

In practice, an assistant is just an *instance* of a graph with a specific configuration. Because of this, multiple assistants can reference the same graph but can contain different configurations, such as prompts, models, and other graph configuration options. The LangGraph Cloud API provides several endpoints for creating and managing assistants. See the [API reference](../cloud/reference/api/api_ref.html) and [this how-to](#docscloudhow-tosconfiguration_cloudmd) for more details on how to create assistants.

### Versioning Assistants

Once you've created an assistant, you can save and version it to track changes to the configuration over time. You can think about this at three levels:

1) The graph lays out the general agent application logic 
2) The agent configuration options represent parameters that can be changed 
3) Assistant versions save and track specific settings of the agent configuration options 

For example, let's imagine you have a general writing agent. You have created a general graph architecture that works well for writing. However, there are different types of writing, e.g. blogs vs tweets. In order to get the best performance on each use case, you need to make some minor changes to the models and prompts used. In this setup, you could create an assistant for each use case - one for blog writing and one for tweeting. These would share the same graph structure, but they may use different models and different prompts. Read [this how-to](#docscloudhow-tosassistant_versioningmd) to learn how you can use assistant versioning through both the [Studio](#docsconceptslanggraph_studiomd) and the SDK.

![assistant versions](img/assistants.png)


### Resources

For more information on assistants, see the following resources:

- [Assistants how-to guides](#assistants)

<a name="docsconceptsauthmd"></a>

## Authentication & Access Control

LangGraph Platform provides a flexible authentication and authorization system that can integrate with most authentication schemes.

!!! note "Python only"

    We currently only support custom authentication and authorization in Python deployments with `langgraph-api>=0.0.11`. Support for LangGraph.JS will be added soon.

### Core Concepts

#### Authentication vs Authorization

While often used interchangeably, these terms represent distinct security concepts:

- [**Authentication**](#authentication) ("AuthN") verifies _who_ you are. This runs as middleware for every request.
- [**Authorization**](#authorization) ("AuthZ") determines _what you can do_. This validates the user's privileges and roles on a per-resource basis.

In LangGraph Platform, authentication is handled by your [`@auth.authenticate`](#langgraph_sdk.auth.Auth.authenticate) handler, and authorization is handled by your [`@auth.on`](#langgraph_sdk.auth.Auth.on) handlers.

### Default Security Models

LangGraph Platform provides different security defaults:

#### LangGraph Cloud

- Uses LangSmith API keys by default
- Requires valid API key in `x-api-key` header
- Can be customized with your auth handler

#### Self-Hosted

- No default authentication
- Complete flexibility to implement your security model
- You control all aspects of authentication and authorization

### System Architecture

A typical authentication setup involves three main components:

1. **Authentication Provider** (Identity Provider/IdP)

    * A dedicated service that manages user identities and credentials
    * Handles user registration, login, password resets, etc.
    * Issues tokens (JWT, session tokens, etc.) after successful authentication
    * Examples: Auth0, Supabase Auth, Okta, or your own auth server

2. **LangGraph Backend** (Resource Server)

    * Your LangGraph application that contains business logic and protected resources
    * Validates tokens with the auth provider
    * Enforces access control based on user identity and permissions
    * Doesn't store user credentials directly

3. **Client Application** (Frontend)

    * Web app, mobile app, or API client
    * Collects time-sensitive user credentials and sends to auth provider
    * Receives tokens from auth provider
    * Includes these tokens in requests to LangGraph backend

Here's how these components typically interact:

```mermaid
sequenceDiagram
    participant Client as Client App
    participant Auth as Auth Provider
    participant LG as LangGraph Backend

    Client->>Auth: 1. Login (username/password)
    Auth-->>Client: 2. Return token
    Client->>LG: 3. Request with token
    Note over LG: 4. Validate token (@auth.authenticate)
    LG-->>Auth:  5. Fetch user info
    Auth-->>LG: 6. Confirm validity
    Note over LG: 7. Apply access control (@auth.on.*)
    LG-->>Client: 8. Return resources
```

Your [`@auth.authenticate`](#langgraph_sdk.auth.Auth.authenticate) handler in LangGraph handles steps 4-6, while your [`@auth.on`](#langgraph_sdk.auth.Auth.on) handlers implement step 7.

### Authentication

Authentication in LangGraph runs as middleware on every request. Your [`@auth.authenticate`](#langgraph_sdk.auth.Auth.authenticate) handler receives request information and should:

1. Validate the credentials
2. Return [user info](#langgraph_sdk.auth.types.MinimalUserDict) containing the user's identity and user information if valid
3. Raise an [HTTP exception](#langgraph_sdk.auth.exceptions.HTTPException) or AssertionError if invalid

```python
from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    # Validate credentials (e.g., API key, JWT token)
    api_key = headers.get("x-api-key")
    if not api_key or not is_valid_key(api_key):
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    # Return user info - only identity and is_authenticated are required
    # Add any additional fields you need for authorization
    return {
        "identity": "user-123",        # Required: unique user identifier
        "is_authenticated": True,      # Optional: assumed True by default
        "permissions": ["read", "write"] # Optional: for permission-based auth
        # You can add more custom fields if you want to implement other auth patterns
        "role": "admin",
        "org_id": "org-456"

    }
```

The returned user information is available:

- To your authorization handlers via [`ctx.user`](#langgraph_sdk.auth.types.AuthContext)
- In your application via `config["configuration"]["langgraph_auth_user"]`

??? tip "Supported Parameters"

    The [`@auth.authenticate`](#langgraph_sdk.auth.Auth.authenticate) handler can accept any of the following parameters by name:

    * request (Request): The raw ASGI request object
    * body (dict): The parsed request body
    * path (str): The request path, e.g., "/threads/abcd-1234-abcd-1234/runs/abcd-1234-abcd-1234/stream"
    * method (str): The HTTP method, e.g., "GET"
    * path_params (dict[str, str]): URL path parameters, e.g., {"thread_id": "abcd-1234-abcd-1234", "run_id": "abcd-1234-abcd-1234"}
    * query_params (dict[str, str]): URL query parameters, e.g., {"stream": "true"}
    * headers (dict[bytes, bytes]): Request headers
    * authorization (str | None): The Authorization header value (e.g., "Bearer <token>")
    
    In many of our tutorials, we will just show the "authorization" parameter to be concise, but you can opt to accept more information as needed
    to implement your custom authentication scheme.

### Authorization

After authentication, LangGraph calls your [`@auth.on`](#langgraph_sdk.auth.Auth.on) handlers to control access to specific resources (e.g., threads, assistants, crons). These handlers can:

1. Add metadata to be saved during resource creation by mutating the `value["metadata"]` dictionary directly. See the [supported actions table](##supported-actions) for the list of types the value can take for each action.
2. Filter resources by metadata during search/list or read operations by returning a [filter dictionary](#filter-operations).
3. Raise an HTTP exception if access is denied.

If you want to just implement simple user-scoped access control, you can use a single [`@auth.on`](#langgraph_sdk.auth.Auth.on) handler for all resources and actions. If you want to have different control depending on the resource and action, you can use [resource-specific handlers](#resource-specific-handlers). See the [Supported Resources](#supported-resources) section for a full list of the resources that support access control.

```python
@auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,
    value: dict  # The payload being sent to this access method
) -> dict:  # Returns a filter dict that restricts access to resources
    """Authorize all access to threads, runs, crons, and assistants.

    This handler does two things:
        - Adds a value to resource metadata (to persist with the resource so it can be filtered later)
        - Returns a filter (to restrict access to existing resources)

    Args:
        ctx: Authentication context containing user info, permissions, the path, and
        value: The request payload sent to the endpoint. For creation
              operations, this contains the resource parameters. For read
              operations, this contains the resource being accessed.

    Returns:
        A filter dictionary that LangGraph uses to restrict access to resources.
        See [Filter Operations](#filter-operations) for supported operators.
    """
    # Create filter to restrict access to just this user's resources
    filters = {"owner": ctx.user.identity}

    # Get or create the metadata dictionary in the payload
    # This is where we store persistent info about the resource
    metadata = value.setdefault("metadata", {})

    # Add owner to metadata - if this is a create or update operation,
    # this information will be saved with the resource
    # So we can filter by it later in read operations
    metadata.update(filters)

    # Return filters to restrict access
    # These filters are applied to ALL operations (create, read, update, search, etc.)
    # to ensure users can only access their own resources
    return filters
```

#### Resource-Specific Handlers {#resource-specific-handlers}

You can register handlers for specific resources and actions by chaining the resource and action names together with the [`@auth.on`](#langgraph_sdk.auth.Auth.on) decorator.
When a request is made, the most specific handler that matches that resource and action is called. Below is an example of how to register handlers for specific resources and actions. For the following setup:

1. Authenticated users are able to create threads, read thread, create runs on threads
2. Only users with the "assistants:create" permission are allowed to create new assistants
3. All other endpoints (e.g., e.g., delete assistant, crons, store) are disabled for all users.

!!! tip "Supported Handlers"

    For a full list of supported resources and actions, see the [Supported Resources](#supported-resources) section below.

```python
## Generic / global handler catches calls that aren't handled by more specific handlers
@auth.on
async def reject_unhandled_requests(ctx: Auth.types.AuthContext, value: Any) -> False:
    print(f"Request to {ctx.path} by {ctx.user.identity}")
    raise Auth.exceptions.HTTPException(
        status_code=403,
        detail="Forbidden"
    )

## Matches the "thread" resource and all actions - create, read, update, delete, search
## Since this is **more specific** than the generic @auth.on handler, it will take precedence
## over the generic handler for all actions on the "threads" resource
@auth.on.threads
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.threads.create.value
):
    if "write" not in ctx.permissions:
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail="User lacks the required permissions."
        )
    # Setting metadata on the thread being created
    # will ensure that the resource contains an "owner" field
    # Then any time a user tries to access this thread or runs within the thread,
    # we can filter by owner
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}

## Thread creation. This will match only on thread create actions
## Since this is **more specific** than both the generic @auth.on handler and the @auth.on.threads handler,
## it will take precedence for any "create" actions on the "threads" resources
@auth.on.threads.create
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.threads.create.value
):
    # Setting metadata on the thread being created
    # will ensure that the resource contains an "owner" field
    # Then any time a user tries to access this thread or runs within the thread,
    # we can filter by owner
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}

## Reading a thread. Since this is also more specific than the generic @auth.on handler, and the @auth.on.threads handler,
## it will take precedence for any "read" actions on the "threads" resource
@auth.on.threads.read
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.threads.read.value
):
    # Since we are reading (and not creating) a thread,
    # we don't need to set metadata. We just need to
    # return a filter to ensure users can only see their own threads
    return {"owner": ctx.user.identity}

## Run creation, streaming, updates, etc.
## This takes precedenceover the generic @auth.on handler and the @auth.on.threads handler
@auth.on.threads.create_run
async def on_run_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.threads.create_run.value
):
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    # Inherit thread's access control
    return {"owner": ctx.user.identity}

## Assistant creation
@auth.on.assistants.create
async def on_assistant_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.assistants.create.value
):
    if "assistants:create" not in ctx.permissions:
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail="User lacks the required permissions."
        )
```

Notice that we are mixing global and resource-specific handlers in the above example. Since each request is handled by the most specific handler, a request to create a `thread` would match the `on_thread_create` handler but NOT the `reject_unhandled_requests` handler. A request to `update` a thread, however would be handled by the global handler, since we don't have a more specific handler for that resource and action. Requests to create, update, 

#### Filter Operations {#filter-operations}

Authorization handlers can return `None`, a boolean, or a filter dictionary.
- `None` and `True` mean "authorize access to all underling resources"
- `False` means "deny access to all underling resources (raises a 403 exception)"
- A metadata filter dictionary will restrict access to resources

A filter dictionary is a dictionary with keys that match the resource metadata. It supports three operators:

- The default value is a shorthand for exact match, or "$eq", below. For example, `{"owner": user_id}` will include only resources with metadata containing `{"owner": user_id}`
- `$eq`: Exact match (e.g., `{"owner": {"$eq": user_id}}`) - this is equivalent to the shorthand above, `{"owner": user_id}`
- `$contains`: List membership (e.g., `{"allowed_users": {"$contains": user_id}}`) The value here must be an element of the list. The metadata in the stored resource must be a list/container type.

A dictionary with multiple keys is treated using a logical `AND` filter. For example, `{"owner": org_id, "allowed_users": {"$contains": user_id}}` will only match resources with metadata whose "owner" is `org_id` and whose "allowed_users" list contains `user_id`.
See the reference [here](#langgraph_sdk.auth.types.FilterType) for more information.

### Common Access Patterns

Here are some typical authorization patterns:

#### Single-Owner Resources

This common pattern lets you scope all threads, assistants, crons, and runs to a single user. It's useful for common single-user use cases like regular chatbot-style apps.

```python
@auth.on
async def owner_only(ctx: Auth.types.AuthContext, value: dict):
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}
```

#### Permission-based Access

This pattern lets you control access based on **permissions**. It's useful if you want certain roles to have broader or more restricted access to resources.

```python
## In your auth handler:
@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    ...
    return {
        "identity": "user-123",
        "is_authenticated": True,
        "permissions": ["threads:write", "threads:read"]  # Define permissions in auth
    }

def _default(ctx: Auth.types.AuthContext, value: dict):
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}

@auth.on.threads.create
async def create_thread(ctx: Auth.types.AuthContext, value: dict):
    if "threads:write" not in ctx.permissions:
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail="Unauthorized"
        )
    return _default(ctx, value)


@auth.on.threads.read
async def rbac_create(ctx: Auth.types.AuthContext, value: dict):
    if "threads:read" not in ctx.permissions and "threads:write" not in ctx.permissions:
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail="Unauthorized"
        )
    return _default(ctx, value)
```

### Supported Resources

LangGraph provides three levels of authorization handlers, from most general to most specific:

1. **Global Handler** (`@auth.on`): Matches all resources and actions
2. **Resource Handler** (e.g., `@auth.on.threads`, `@auth.on.assistants`, `@auth.on.crons`): Matches all actions for a specific resource
3. **Action Handler** (e.g., `@auth.on.threads.create`, `@auth.on.threads.read`): Matches a specific action on a specific resource

The most specific matching handler will be used. For example, `@auth.on.threads.create` takes precedence over `@auth.on.threads` for thread creation.
If a more specific handler is registered, the more general handler will not be called for that resource and action.

???+ tip "Type Safety"
    Each handler has type hints available for its `value` parameter at `Auth.types.on.<resource>.<action>.value`. For example:
    ```python
    @auth.on.threads.create
    async def on_thread_create(
        ctx: Auth.types.AuthContext,
        value: Auth.types.on.threads.create.value  # Specific type for thread creation
    ):
        ...
    
    @auth.on.threads
    async def on_threads(
        ctx: Auth.types.AuthContext,
        value: Auth.types.on.threads.value  # Union type of all thread actions
    ):
        ...
    
    @auth.on
    async def on_all(
        ctx: Auth.types.AuthContext,
        value: dict  # Union type of all possible actions
    ):
        ...
    ```
    More specific handlers provide better type hints since they handle fewer action types.

##### Supported actions and types {#supported-actions}
Here are all the supported action handlers:

| Resource | Handler | Description | Value Type |
|----------|---------|-------------|------------|
| **Threads** | `@auth.on.threads.create` | Thread creation | [`ThreadsCreate`](#langgraph_sdk.auth.types.ThreadsCreate) |
| | `@auth.on.threads.read` | Thread retrieval | [`ThreadsRead`](#langgraph_sdk.auth.types.ThreadsRead) |
| | `@auth.on.threads.update` | Thread updates | [`ThreadsUpdate`](#langgraph_sdk.auth.types.ThreadsUpdate) |
| | `@auth.on.threads.delete` | Thread deletion | [`ThreadsDelete`](#langgraph_sdk.auth.types.ThreadsDelete) |
| | `@auth.on.threads.search` | Listing threads | [`ThreadsSearch`](#langgraph_sdk.auth.types.ThreadsSearch) |
| | `@auth.on.threads.create_run` | Creating or updating a run | [`RunsCreate`](#langgraph_sdk.auth.types.RunsCreate) |
| **Assistants** | `@auth.on.assistants.create` | Assistant creation | [`AssistantsCreate`](#langgraph_sdk.auth.types.AssistantsCreate) |
| | `@auth.on.assistants.read` | Assistant retrieval | [`AssistantsRead`](#langgraph_sdk.auth.types.AssistantsRead) |
| | `@auth.on.assistants.update` | Assistant updates | [`AssistantsUpdate`](#langgraph_sdk.auth.types.AssistantsUpdate) |
| | `@auth.on.assistants.delete` | Assistant deletion | [`AssistantsDelete`](#langgraph_sdk.auth.types.AssistantsDelete) |
| | `@auth.on.assistants.search` | Listing assistants | [`AssistantsSearch`](#langgraph_sdk.auth.types.AssistantsSearch) |
| **Crons** | `@auth.on.crons.create` | Cron job creation | [`CronsCreate`](#langgraph_sdk.auth.types.CronsCreate) |
| | `@auth.on.crons.read` | Cron job retrieval | [`CronsRead`](#langgraph_sdk.auth.types.CronsRead) |
| | `@auth.on.crons.update` | Cron job updates | [`CronsUpdate`](#langgraph_sdk.auth.types.CronsUpdate) |
| | `@auth.on.crons.delete` | Cron job deletion | [`CronsDelete`](#langgraph_sdk.auth.types.CronsDelete) |
| | `@auth.on.crons.search` | Listing cron jobs | [`CronsSearch`](#langgraph_sdk.auth.types.CronsSearch) |

???+ note "About Runs"
    Runs are scoped to their parent thread for access control. This means permissions are typically inherited from the thread, reflecting the conversational nature of the data model. All run operations (reading, listing) except creation are controlled by the thread's handlers.
    There is a specific `create_run` handler for creating new runs because it had more arguments that you can view in the handler.


### Next Steps

For implementation details:

- Check out the introductory tutorial on [setting up authentication](#docstutorialsauthgetting_startedmd)
- See the how-to guide on implementing a [custom auth handlers](#docshow-tosauthcustom_authmd)


<a name="docsconceptsbreakpointsmd"></a>

## Breakpoints

Breakpoints pause graph execution at specific points and enable stepping through execution step by step. Breakpoints are powered by LangGraph's [**persistence layer**](#docsconceptspersistencemd), which saves the state after each graph step. Breakpoints can also be used to enable [**human-in-the-loop**](#docsconceptshuman_in_the_loopmd) workflows, though we recommend using the [`interrupt` function](#interrupt) for this purpose.

### Requirements

To use breakpoints, you will need to:

1. [**Specify a checkpointer**](#checkpoints) to save the graph state after each step.
2. [**Set breakpoints**](#setting-breakpoints) to specify where execution should pause.
3. **Run the graph** with a [**thread ID**](#threads) to pause execution at the breakpoint.
4. **Resume execution** using `invoke`/`ainvoke`/`stream`/`astream` (see [**The `Command` primitive**](#the-command-primitive)).

### Setting breakpoints

There are two places where you can set breakpoints:

1. **Before** or **after** a node executes by setting breakpoints at **compile time** or **run time**. We call these [**static breakpoints**](#static-breakpoints).
2. **Inside** a node using the [`NodeInterrupt` exception](#nodeinterrupt-exception).
 
#### Static breakpoints

Static breakpoints are triggered either **before** or **after** a node executes. You can set static breakpoints by specifying `interrupt_before` and `interrupt_after` at **"compile" time** or **run time**.

=== "Compile time"

    ```python
    graph = graph_builder.compile(
        interrupt_before=["node_a"], 
        interrupt_after=["node_b", "node_c"],
        checkpointer=..., # Specify a checkpointer
    )

    thread_config = {
        "configurable": {
            "thread_id": "some_thread"
        }
    }

    # Run the graph until the breakpoint
    graph.invoke(inputs, config=thread_config)

    # Optionally update the graph state based on user input
    graph.update_state(update, config=thread_config)

    # Resume the graph
    graph.invoke(None, config=thread_config)
    ```

=== "Run time"

    ```python
    graph.invoke(
        inputs, 
        config={"configurable": {"thread_id": "some_thread"}}, 
        interrupt_before=["node_a"], 
        interrupt_after=["node_b", "node_c"]
    )

    thread_config = {
        "configurable": {
            "thread_id": "some_thread"
        }
    }

    # Run the graph until the breakpoint
    graph.invoke(inputs, config=thread_config)

    # Optionally update the graph state based on user input
    graph.update_state(update, config=thread_config)

    # Resume the graph
    graph.invoke(None, config=thread_config)
    ```

    !!! note

        You cannot set static breakpoints at runtime for **sub-graphs**.
        If you have a sub-graph, you must set the breakpoints at compilation time.

Static breakpoints can be especially useful for debugging if you want to step through the graph execution one
node at a time or if you want to pause the graph execution at specific nodes.

#### `NodeInterrupt` exception

We recommend that you [**use the `interrupt` function instead**](#the-interrupt-function) of the `NodeInterrupt` exception if you're trying to implement
[human-in-the-loop](#docsconceptshuman_in_the_loopmd) workflows. The `interrupt` function is easier to use and more flexible.

??? node "`NodeInterrupt` exception"

    The developer can define some *condition* that must be met for a breakpoint to be triggered. This concept of [dynamic breakpoints](#dynamic-breakpoints) is useful when the developer wants to halt the graph under *a particular condition*. This uses a `NodeInterrupt`, which is a special type of exception that can be raised from within a node based upon some condition. As an example, we can define a dynamic breakpoint that triggers when the `input` is longer than 5 characters.

    ```python
    def my_node(state: State) -> State:
        if len(state['input']) > 5:
            raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")

        return state
    ```


    Let's assume we run the graph with an input that triggers the dynamic breakpoint and then attempt to resume the graph execution simply by passing in `None` for the input.

    ```python
    # Attempt to continue the graph execution with no change to state after we hit the dynamic breakpoint 
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    ```

    The graph will *interrupt* again because this node will be *re-run* with the same graph state. We need to change the graph state such that the condition that triggers the dynamic breakpoint is no longer met. So, we can simply edit the graph state to an input that meets the condition of our dynamic breakpoint (< 5 characters) and re-run the node.

    ```python 
    # Update the state to pass the dynamic breakpoint
    graph.update_state(config=thread_config, values={"input": "foo"})
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    ```

    Alternatively, what if we want to keep our current input and skip the node (`my_node`) that performs the check? To do this, we can simply perform the graph update with `as_node="my_node"` and pass in `None` for the values. This will make no update the graph state, but run the update as `my_node`, effectively skipping the node and bypassing the dynamic breakpoint.

    ```python
    # This update will skip the node `my_node` altogether
    graph.update_state(config=thread_config, values=None, as_node="my_node")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    ```

### Additional Resources ­ƒôÜ

- [**Conceptual Guide: Persistence**](#docsconceptspersistencemd): Read the persistence guide for more context about persistence.
- [**Conceptual Guide: Human-in-the-loop**](#docsconceptshuman_in_the_loopmd): Read the human-in-the-loop guide for more context on integrating human feedback into LangGraph applications using breakpoints.
- [**How to View and Update Past Graph State**](../how-tos/human_in_the_loop/time-travel.ipynb): Step-by-step instructions for working with graph state that demonstrate the **replay** and **fork** actions.

<a name="docsconceptsbring_your_own_cloudmd"></a>

## Bring Your Own Cloud (BYOC)

!!! note Prerequisites

    - [LangGraph Platform](#docsconceptslanggraph_platformmd)
    - [Deployment Options](#docsconceptsdeployment_optionsmd)

### Architecture

Split control plane (hosted by us) and data plane (hosted by you, managed by us).

|                             | Control Plane                   | Data Plane                                    |
|-----------------------------|---------------------------------|-----------------------------------------------|
| What it does                | Manages deployments, revisions. | Runs your LangGraph graphs, stores your data. |
| Where it is hosted          | LangChain Cloud account         | Your cloud account                            |
| Who provisions and monitors | LangChain                       | LangChain                                     |

LangChain has no direct access to the resources created in your cloud account, and can only interact with them via AWS APIs. Your data never leaves your cloud account / VPC at rest or in transit.

![Architecture](img/byoc_architecture.png)

### Requirements

- YouÔÇÖre using AWS already.
- You use `langgraph-cli` and/or [LangGraph Studio](#docsconceptslanggraph_studiomd) app to test graph locally.
- You use `langgraph build` command to build image and then push it to your AWS ECR repository (`docker push`).

### How it works

- We provide you a [Terraform module](https://github.com/langchain-ai/terraform/tree/main/modules/langgraph_cloud_setup) which you run to set up our requirements
    1. Creates an AWS role (which our control plane will later assume to provision and monitor resources)
        - https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonVPCReadOnlyAccess.html
            - Read VPCS to find subnets
        - https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonECS_FullAccess.html
            - Used to create/delete ECS resources for your LangGraph Cloud instances
        - https://docs.aws.amazon.com/aws-managed-policy/latest/reference/SecretsManagerReadWrite.html
            - Create secrets for your ECS resources
        - https://docs.aws.amazon.com/aws-managed-policy/latest/reference/CloudWatchReadOnlyAccess.html
            - Read CloudWatch metrics/logs to monitor your instances/push deployment logs
        - https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonRDSFullAccess.html
            - Provision `RDS` instances for your LangGraph Cloud instances
    2. Either
        - Tags an existing vpc / subnets as `langgraph-cloud-enabled`
        - Creates a new vpc and subnets and tags them as `langgraph-cloud-enabled`
- You create a LangGraph Cloud Project in `smith.langchain.com` providing
    - the ID of the AWS role created in the step above
    - the AWS ECR repo to pull the service image from
- We provision the resources in your cloud account using the role above
- We monitor those resources to ensure uptime and recovery from errors

Notes for customers using [self-hosted LangSmith](https://docs.smith.langchain.com/self_hosting):

- Creation of new LangGraph Cloud projects and revisions currently needs to be done on smith.langchain.com.
- You can however set up the project to trace to your self-hosted LangSmith instance if desired


<a name="docsconceptsdeployment_optionsmd"></a>

## Deployment Options

!!! info "Prerequisites"

    - [LangGraph Platform](#docsconceptslanggraph_platformmd)
    - [LangGraph Server](#docsconceptslanggraph_servermd)
    - [LangGraph Platform Plans](#docsconceptsplansmd)

### Overview

There are 4 main options for deploying with the LangGraph Platform:

1. **[Self-Hosted Lite](#self-hosted-lite)**: Available for all plans.

2. **[Self-Hosted Enterprise](#self-hosted-enterprise)**: Available for the **Enterprise** plan.

3. **[Cloud SaaS](#cloud-saas)**: Available for **Plus** and **Enterprise** plans.

4. **[Bring Your Own Cloud](#bring-your-own-cloud)**: Available only for **Enterprise** plans and **only on AWS**.

Please see the [LangGraph Platform Plans](#docsconceptsplansmd) for more information on the different plans.

The guide below will explain the differences between the deployment options.

### Self-Hosted Enterprise

!!! important

    The Self-Hosted Enterprise version is only available for the **Enterprise** plan.

With a Self-Hosted Enterprise deployment, you are responsible for managing the infrastructure, including setting up and maintaining required databases and Redis instances.

YouÔÇÖll build a Docker image using the [LangGraph CLI](#docsconceptslanggraph_climd), which can then be deployed on your own infrastructure.

For more information, please see:

* [Self-Hosted conceptual guide](#docsconceptsself_hostedmd)
* [Self-Hosted Deployment how-to guide](#docshow-tosdeploy-self-hostedmd)

### Self-Hosted Lite

!!! important

    The Self-Hosted Lite version is available for all plans.

The Self-Hosted Lite deployment option is a free (up to 1 million nodes executed), limited version of LangGraph Platform that you can run locally or in a self-hosted manner.

With a Self-Hosted Lite deployment, you are responsible for managing the infrastructure, including setting up and maintaining required databases and Redis instances.

YouÔÇÖll build a Docker image using the [LangGraph CLI](#docsconceptslanggraph_climd), which can then be deployed on your own infrastructure.


For more information, please see:

* [Self-Hosted conceptual guide](#docsconceptsself_hostedmd)
* [Self-Hosted deployment how-to guide](#docshow-tosdeploy-self-hostedmd)

### Cloud SaaS

!!! important

    The Cloud SaaS version of LangGraph Platform is only available for **Plus** and **Enterprise** plans.


The [Cloud SaaS](#docsconceptslanggraph_cloudmd) version of LangGraph Platform is hosted as part of [LangSmith](https://smith.langchain.com/).

The Cloud SaaS version of LangGraph Platform provides a simple way to deploy and manage your LangGraph applications.

This deployment option provides an integration with GitHub, allowing you to deploy code from any of your repositories on GitHub.

For more information, please see:

* [Cloud SaaS Conceptual Guide](#docsconceptslanggraph_cloudmd)
* [How to deploy to Cloud SaaS](#docsclouddeploymentcloudmd)


### Bring Your Own Cloud

!!! important

    The Bring Your Own Cloud version of LangGraph Platform is only available for **Enterprise** plans.


This combines the best of both worlds for Cloud and Self-Hosted. We manage the infrastructure, so you don't have to, but the infrastructure all runs within your cloud. This is currently only available on AWS.

For more information please see:

* [Bring Your Own Cloud Conceptual Guide](#docsconceptsbring_your_own_cloudmd)

### Related

For more information, please see:

* [LangGraph Platform plans](#docsconceptsplansmd)
* [LangGraph Platform pricing](https://www.langchain.com/langgraph-platform-pricing)
* [Deployment how-to guides](#deployment)


<a name="docsconceptsdouble_textingmd"></a>

## Double Texting

!!! info "Prerequisites"
    - [LangGraph Server](#docsconceptslanggraph_servermd)

Many times users might interact with your graph in unintended ways. 
For instance, a user may send one message and before the graph has finished running send a second message. 
More generally, users may invoke the graph a second time before the first run has finished.
We call this "double texting".

Currently, LangGraph only addresses this as part of [LangGraph Platform](#docsconceptslanggraph_platformmd), not in the open source.
The reason for this is that in order to handle this we need to know how the graph is deployed, and since LangGraph Platform deals with deployment the logic needs to live there.
If you do not want to use LangGraph Platform, we describe the options we have implemented in detail below.

![](img/double_texting.png)

### Reject

This is the simplest option, this just rejects any follow-up runs and does not allow double texting. 
See the [how-to guide](#docscloudhow-tosreject_concurrentmd) for configuring the reject double text option.

### Enqueue

This is a relatively simple option which continues the first run until it completes the whole run, then sends the new input as a separate run. 
See the [how-to guide](#docscloudhow-tosenqueue_concurrentmd) for configuring the enqueue double text option.

### Interrupt

This option interrupts the current execution but saves all the work done up until that point. 
It then inserts the user input and continues from there. 

If you enable this option, your graph should be able to handle weird edge cases that may arise. 
For example, you could have called a tool but not yet gotten back a result from running that tool.
You may need to remove that tool call in order to not have a dangling tool call.

See the [how-to guide](#docscloudhow-tosinterrupt_concurrentmd) for configuring the interrupt double text option.

### Rollback

This option interrupts the current execution AND rolls back all work done up until that point, including the original run input. It then sends the new user input in, basically as if it was the original input.

See the [how-to guide](#docscloudhow-tosrollback_concurrentmd) for configuring the rollback double text option.


<a name="docsconceptsfaqmd"></a>

## FAQ

Common questions and their answers!

### Do I need to use LangChain to use LangGraph? WhatÔÇÖs the difference?

No. LangGraph is an orchestration framework for complex agentic systems and is more low-level and controllable than LangChain agents. LangChain provides a standard interface to interact with models and other components, useful for straight-forward chains and retrieval flows.

### How is LangGraph different from other agent frameworks?

Other agentic frameworks can work for simple, generic tasks but fall short for complex tasks bespoke to a companyÔÇÖs needs. LangGraph provides a more expressive framework to handle companiesÔÇÖ unique tasks without restricting users to a single black-box cognitive architecture.

### Does LangGraph impact the performance of my app?

LangGraph will not add any overhead to your code and is specifically designed with streaming workflows in mind.

### Is LangGraph open source? Is it free?

Yes. LangGraph is an MIT-licensed open-source library and is free to use.

### How are LangGraph and LangGraph Platform different?

LangGraph is a stateful, orchestration framework that brings added control to agent workflows. LangGraph Platform is a service for deploying and scaling LangGraph applications, with an opinionated API for building agent UXs, plus an integrated developer studio.

| Features            | LangGraph (open source)                                   | LangGraph Platform                                                                                     |
|---------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Description         | Stateful orchestration framework for agentic applications | Scalable infrastructure for deploying LangGraph applications                                           |
| SDKs                | Python and JavaScript                                     | Python and JavaScript                                                                                  |
| HTTP APIs           | None                                                      | Yes - useful for retrieving & updating state or long-term memory, or creating a configurable assistant |
| Streaming           | Basic                                                     | Dedicated mode for token-by-token messages                                                             |
| Checkpointer        | Community contributed                                     | Supported out-of-the-box                                                                               |
| Persistence Layer   | Self-managed                                              | Managed Postgres with efficient storage                                                                |
| Deployment          | Self-managed                                              | ÔÇó Cloud SaaS <br> ÔÇó Free self-hosted <br> ÔÇó Enterprise (BYOC or paid self-hosted)                      |
| Scalability         | Self-managed                                              | Auto-scaling of task queues and servers                                                                |
| Fault-tolerance     | Self-managed                                              | Automated retries                                                                                      |
| Concurrency Control | Simple threading                                          | Supports double-texting                                                                                |
| Scheduling          | None                                                      | Cron scheduling                                                                                        |
| Monitoring          | None                                                      | Integrated with LangSmith for observability                                                            |
| IDE integration     | LangGraph Studio for Desktop                              | LangGraph Studio for Desktop & Cloud                                                                   |

### What are my deployment options for LangGraph Platform?

We currently have the following deployment options for LangGraph applications:

- [ÔÇìSelf-Hosted Lite](#self-hosted-lite): A free (up to 1M nodes executed), limited version of LangGraph Platform that you can run locally or in a self-hosted manner. This version requires a LangSmith API key and logs all usage to LangSmith. Fewer features are available than in paid plans.
- [Cloud SaaS](#cloud-saas): Fully managed and hosted as part of LangSmith, with automatic updates and zero maintenance.
- [ÔÇìBring Your Own Cloud (BYOC)](#bring-your-own-cloud): Deploy LangGraph Platform within your VPC, provisioned and run as a service. Keep data in your environment while outsourcing the management of the service.
- [Self-Hosted Enterprise](#self-hosted-enterprise): Deploy LangGraph entirely on your own infrastructure.

### Is LangGraph Platform open source?

No. LangGraph Platform is proprietary software.

There is a free, self-hosted version of LangGraph Platform with access to basic features. The Cloud SaaS deployment option is free while in beta, but will eventually be a paid service. We will always give ample notice before charging for a service and reward our early adopters with preferential pricing. The Bring Your Own Cloud (BYOC) and Self-Hosted Enterprise options are also paid services. [Contact our sales team](https://www.langchain.com/contact-sales) to learn more.

For more information, see our [LangGraph Platform pricing page](https://www.langchain.com/pricing-langgraph-platform).

### Does LangGraph work with LLMs that don't support tool calling?

Yes! You can use LangGraph with any LLMs. The main reason we use LLMs that support tool calling is that this is often the most convenient way to have the LLM make its decision about what to do. If your LLM does not support tool calling, you can still use it - you just need to write a bit of logic to convert the raw LLM string response to a decision about what to do.

### Does LangGraph work with OSS LLMs?

Yes! LangGraph is totally ambivalent to what LLMs are used under the hood. The main reason we use closed LLMs in most of the tutorials is that they seamlessly support tool calling, while OSS LLMs often don't. But tool calling is not necessary (see [this section](#does-langgraph-work-with-llms-that-dont-support-tool-calling)) so you can totally use LangGraph with OSS LLMs.


<a name="docsconceptshigh_levelmd"></a>

## Why LangGraph?

LLMs are extremely powerful, particularly when connected to other systems such as a retriever or APIs. This is why many LLM applications use a control flow of steps before and / or after LLM calls. As an example [RAG](https://github.com/langchain-ai/rag-from-scratch) performs retrieval of relevant documents to a question, and passes those documents to an LLM in order to ground the response. Often a control flow of steps before and / or after an LLM is called a "chain." Chains are a popular paradigm for programming with LLMs and offer a high degree of reliability; the same set of steps runs with each chain invocation.

However, we often want LLM systems that can pick their own control flow! This is one definition of an [agent](https://blog.langchain.dev/what-is-an-agent/): an agent is a system that uses an LLM to decide the control flow of an application. Unlike a chain, an agent gives an LLM some degree of control over the sequence of steps in the application. Examples of using an LLM to decide the control of an application:

- Using an LLM to route between two potential paths
- Using an LLM to decide which of many tools to call
- Using an LLM to decide whether the generated answer is sufficient or more work is need

There are many different types of [agent architectures](https://blog.langchain.dev/what-is-a-cognitive-architecture/) to consider, which give an LLM varying levels of control. On one extreme, a router allows an LLM to select a single step from a specified set of options and, on the other extreme, a fully autonomous long-running agent may have complete freedom to select any sequence of steps that it wants for a given problem. 

![Agent Types](img/agent_types.png)

Several concepts are utilized in many agent architectures:

- [Tool calling](#tool-calling): this is often how LLMs make decisions
- Action taking: often times, the LLMs' outputs are used as the input to an action
- [Memory](#memory): reliable systems need to have knowledge of things that occurred
- [Planning](#planning): planning steps (either explicit or implicit) are useful for ensuring that the LLM, when making decisions, makes them in the highest fidelity way.

### Challenges

In practice, there is often a trade-off between control and reliability. As we give LLMs more control, the application often become less reliable. This can be due to factors such as LLM non-determinism and / or errors in selecting tools (or steps) that the agent uses (takes).

![Agent Challenge](img/challenge.png)

### Core Principles

The motivation of LangGraph is to help bend the curve, preserving higher reliability as we give the agent more control over the application. We'll outline a few specific pillars of LangGraph that make it well suited for building reliable agents. 

![Langgraph](img/langgraph.png)

**Controllability**

LangGraph gives the developer a high degree of [control](#controllability) by expressing the flow of the application as a set of nodes and edges. All nodes can access and modify a common state (memory). The control flow of the application can set using edges that connect nodes, either deterministically or via conditional logic. 

**Persistence**

LangGraph gives the developer many options for [persisting](#persistence) graph state using short-term or long-term (e.g., via a database) memory. 

**Human-in-the-Loop**

The persistence layer enables several different [human-in-the-loop](#human-in-the-loop) interaction patterns with agents; for example, it's possible to pause an agent, review its state, edit it state, and approve a follow-up step. 

**Streaming**

LangGraph comes with first class support for [streaming](#streaming), which can expose state to the user (or developer) over the course of agent execution. LangGraph supports streaming of both events ([like a tool call being taken](../how-tos/stream-updates.ipynb)) as well as of [tokens that an LLM may emit](../how-tos/streaming-tokens.ipynb).

### Debugging

Once you've built a graph, you often want to test and debug it. [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file) is a specialized IDE for visualization and debugging of LangGraph applications.

![Langgraph Studio](img/lg_studio.png)

### Deployment

Once you have confidence in your LangGraph application, many developers want an easy path to deployment. [LangGraph Platform](#langgraph-platform) offers a range of options for deploying LangGraph graphs.

<a name="docsconceptshuman_in_the_loopmd"></a>

## Human-in-the-loop

!!! tip "This guide uses the new `interrupt` function."

    As of LangGraph 0.2.57, the recommended way to set breakpoints is using the [`interrupt` function][langgraph.types.interrupt] as it simplifies **human-in-the-loop** patterns.

    If you're looking for the previous version of this conceptual guide, which relied on static breakpoints and `NodeInterrupt` exception, it is available [here](#docsconceptsv0-human-in-the-loopmd). 

A **human-in-the-loop** (or "on-the-loop") workflow integrates human input into automated processes, allowing for decisions, validation, or corrections at key stages. This is especially useful in **LLM-based applications**, where the underlying model may generate occasional inaccuracies. In low-error-tolerance scenarios like compliance, decision-making, or content generation, human involvement ensures reliability by enabling review, correction, or override of model outputs.


### Use cases

Key use cases for **human-in-the-loop** workflows in LLM-based applications include:

1. [**­ƒøá´©Å Reviewing tool calls**](#review-tool-calls): Humans can review, edit, or approve tool calls requested by the LLM before tool execution.
2. **Ô£à Validating LLM outputs**: Humans can review, edit, or approve content generated by the LLM.
3. **­ƒÆí Providing context**: Enable the LLM to explicitly request human input for clarification or additional details or to support multi-turn conversations.

### `interrupt`

The [`interrupt` function][langgraph.types.interrupt] in LangGraph enables human-in-the-loop workflows by pausing the graph at a specific node, presenting information to a human, and resuming the graph with their input. This function is useful for tasks like approvals, edits, or collecting additional input. The [`interrupt` function][langgraph.types.interrupt] is used in conjunction with the [`Command`](#langgraph.types.Command) object to resume the graph with a value provided by the human.

```python
from langgraph.types import interrupt

def human_node(state: State):
    value = interrupt(
        # Any JSON serializable value to surface to the human.
        # For example, a question or a piece of text or a set of keys in the state
       {
          "text_to_revise": state["some_text"]
       }
    )
    # Update the state with the human's input or route the graph based on the input.
    return {
        "some_text": value
    }

graph = graph_builder.compile(
    checkpointer=checkpointer # Required for `interrupt` to work
)

## Run the graph until the interrupt
thread_config = {"configurable": {"thread_id": "some_id"}}
graph.invoke(some_input, config=thread_config)
    
## Resume the graph with the human's input
graph.invoke(Command(resume=value_from_human), config=thread_config)
```

```pycon
{'some_text': 'Edited text'}
```

!!! warning
      Interrupts are both powerful and ergonomic. However, while they may resemble Python's input() function in terms of developer experience, it's important to note that they do not automatically resume execution from the interruption point. Instead, they rerun the entire node where the interrupt was used.
      For this reason, interrupts are typically best placed at the start of a node or in a dedicated node. Please read the [resuming from an interrupt](#how-does-resuming-from-an-interrupt-work) section for more details. 

??? "Full Code"

      Here's a full example of how to use `interrupt` in a graph, if you'd like
      to see the code in action.

      ```python
      from typing import TypedDict
      import uuid

      from langgraph.checkpoint.memory import MemorySaver
      from langgraph.constants import START
      from langgraph.graph import StateGraph
      from langgraph.types import interrupt, Command

      class State(TypedDict):
         """The graph state."""
         some_text: str

      def human_node(state: State):
         value = interrupt(
            # Any JSON serializable value to surface to the human.
            # For example, a question or a piece of text or a set of keys in the state
            {
               "text_to_revise": state["some_text"]
            }
         )
         return {
            # Update the state with the human's input
            "some_text": value
         }


      # Build the graph
      graph_builder = StateGraph(State)
      # Add the human-node to the graph
      graph_builder.add_node("human_node", human_node)
      graph_builder.add_edge(START, "human_node")

      # A checkpointer is required for `interrupt` to work.
      checkpointer = MemorySaver()
      graph = graph_builder.compile(
         checkpointer=checkpointer
      )

      # Pass a thread ID to the graph to run it.
      thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

      # Using stream() to directly surface the `__interrupt__` information.
      for chunk in graph.stream({"some_text": "Original text"}, config=thread_config):
         print(chunk)

      # Resume using Command
      for chunk in graph.stream(Command(resume="Edited text"), config=thread_config):
         print(chunk)
      ```

      ```pycon
      {'__interrupt__': (
            Interrupt(
               value={'question': 'Please revise the text', 'some_text': 'Original text'}, 
               resumable=True, 
               ns=['human_node:10fe492f-3688-c8c6-0d0a-ec61a43fecd6'], 
               when='during'
            ),
         )
      }
      {'human_node': {'some_text': 'Edited text'}}
      ```

### Requirements

To use `interrupt` in your graph, you need to:

1. [**Specify a checkpointer**](#checkpoints) to save the graph state after each step.
2. **Call `interrupt()`** in the appropriate place. See the [Design Patterns](#design-patterns) section for examples.
3. **Run the graph** with a [**thread ID**](#threads) until the `interrupt` is hit.
4. **Resume execution** using `invoke`/`ainvoke`/`stream`/`astream` (see [**The `Command` primitive**](#the-command-primitive)).

### Design Patterns

There are typically three different **actions** that you can do with a human-in-the-loop workflow:

1. **Approve or Reject**: Pause the graph before a critical step, such as an API call, to review and approve the action. If the action is rejected, you can prevent the graph from executing the step, and potentially take an alternative action. This pattern often involve **routing** the graph based on the human's input.
2. **Edit Graph State**: Pause the graph to review and edit the graph state. This is useful for correcting mistakes or updating the state with additional information. This pattern often involves **updating** the state with the human's input.
3. **Get Input**: Explicitly request human input at a particular step in the graph. This is useful for collecting additional information or context to inform the agent's decision-making process or for supporting **multi-turn conversations**.

Below we show different design patterns that can be implemented using these **actions**.

#### Approve or Reject

<figure markdown="1">
![image](img/human_in_the_loop/approve-or-reject.png){: style="max-height:400px"}
<figcaption>Depending on the human's approval or rejection, the graph can proceed with the action or take an alternative path.</figcaption>
</figure>

Pause the graph before a critical step, such as an API call, to review and approve the action. If the action is rejected, you can prevent the graph from executing the step, and potentially take an alternative action.

```python

from typing import Literal
from langgraph.types import interrupt, Command

def human_approval(state: State) -> Command[Literal["some_node", "another_node"]]:
    is_approved = interrupt(
        {
            "question": "Is this correct?",
            # Surface the output that should be
            # reviewed and approved by the human.
            "llm_output": state["llm_output"]
        }
    )

    if is_approved:
        return Command(goto="some_node")
    else:
        return Command(goto="another_node")

## Add the node to the graph in an appropriate location
## and connect it to the relevant nodes.
graph_builder.add_node("human_approval", human_approval)
graph = graph_builder.compile(checkpointer=checkpointer)

## After running the graph and hitting the interrupt, the graph will pause.
## Resume it with either an approval or rejection.
thread_config = {"configurable": {"thread_id": "some_id"}}
graph.invoke(Command(resume=True), config=thread_config)
```

See [how to review tool calls](../how-tos/human_in_the_loop/review-tool-calls.ipynb) for a more detailed example.

#### Review & Edit State

<figure markdown="1">
![image](img/human_in_the_loop/edit-graph-state-simple.png){: style="max-height:400px"}
<figcaption>A human can review and edit the state of the graph. This is useful for correcting mistakes or updating the state with additional information.
</figcaption>
</figure>

```python
from langgraph.types import interrupt

def human_editing(state: State):
    ...
    result = interrupt(
        # Interrupt information to surface to the client.
        # Can be any JSON serializable value.
        {
            "task": "Review the output from the LLM and make any necessary edits.",
            "llm_generated_summary": state["llm_generated_summary"]
        }
    )

    # Update the state with the edited text
    return {
        "llm_generated_summary": result["edited_text"] 
    }

## Add the node to the graph in an appropriate location
## and connect it to the relevant nodes.
graph_builder.add_node("human_editing", human_editing)
graph = graph_builder.compile(checkpointer=checkpointer)

...

## After running the graph and hitting the interrupt, the graph will pause.
## Resume it with the edited text.
thread_config = {"configurable": {"thread_id": "some_id"}}
graph.invoke(
    Command(resume={"edited_text": "The edited text"}), 
    config=thread_config
)
```

See [How to wait for user input using interrupt](../how-tos/human_in_the_loop/wait-user-input.ipynb) for a more detailed example.

#### Review Tool Calls

<figure markdown="1">
![image](img/human_in_the_loop/tool-call-review.png){: style="max-height:400px"}
<figcaption>A human can review and edit the output from the LLM before proceeding. This is particularly
critical in applications where the tool calls requested by the LLM may be sensitive or require human oversight.
</figcaption>
</figure>

```python
def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
    # This is the value we'll be providing via Command(resume=<human_review>)
    human_review = interrupt(
        {
            "question": "Is this correct?",
            # Surface tool calls for review
            "tool_call": tool_call
        }
    )

    review_action, review_data = human_review

    # Approve the tool call and continue
    if review_action == "continue":
        return Command(goto="run_tool")

    # Modify the tool call manually and then continue
    elif review_action == "update":
        ...
        updated_msg = get_updated_msg(review_data)
        # Remember that to modify an existing message you will need
        # to pass the message with a matching ID.
        return Command(goto="run_tool", update={"messages": [updated_message]})

    # Give natural language feedback, and then pass that back to the agent
    elif review_action == "feedback":
        ...
        feedback_msg = get_feedback_msg(review_data)
        return Command(goto="call_llm", update={"messages": [feedback_msg]})
```

See [how to review tool calls](../how-tos/human_in_the_loop/review-tool-calls.ipynb) for a more detailed example.

#### Multi-turn conversation

<figure markdown="1">
![image](img/human_in_the_loop/multi-turn-conversation.png){: style="max-height:400px"}
<figcaption>A <strong>multi-turn conversation</strong> architecture where an <strong>agent</strong> and <strong>human node</strong> cycle back and forth until the agent decides to hand off the conversation to another agent or another part of the system.
</figcaption>
</figure>

A **multi-turn conversation** involves multiple back-and-forth interactions between an agent and a human, which can allow the agent to gather additional information from the human in a conversational manner.

This design pattern is useful in an LLM application consisting of [multiple agents](#docsconceptsmulti_agentmd). One or more agents may need to carry out multi-turn conversations with a human, where the human provides input or feedback at different stages of the conversation. For simplicity, the agent implementation below is illustrated as a single node, but in reality 
it may be part of a larger graph consisting of multiple nodes and include a conditional edge.

=== "Using a human node per agent"

    In this pattern, each agent has its own human node for collecting user input. 
    This can be achieved by either naming the human nodes with unique names (e.g., "human for agent 1", "human for agent 2") or by
    using subgraphs where a subgraph contains a human node and an agent node.

    ```python
    from langgraph.types import interrupt

    def human_input(state: State):
        human_message = interrupt("human_input")
        return {
            "messages": [
                {
                    "role": "human",
                    "content": human_message
                }
            ]
        }

    def agent(state: State):
        # Agent logic
        ...

    graph_builder.add_node("human_input", human_input)
    graph_builder.add_edge("human_input", "agent")
    graph = graph_builder.compile(checkpointer=checkpointer)

    # After running the graph and hitting the interrupt, the graph will pause.
    # Resume it with the human's input.
    graph.invoke(
        Command(resume="hello!"),
        config=thread_config
    )
    ```


=== "Sharing human node across multiple agents"

    In this pattern, a single human node is used to collect user input for multiple agents. The active agent is determined from the state, so after human input is collected, the graph can route to the correct agent.

    ```python
    from langgraph.types import interrupt

    def human_node(state: MessagesState) -> Command[Literal["agent_1", "agent_2", ...]]:
        """A node for collecting user input."""
        user_input = interrupt(value="Ready for user input.")

        # Determine the **active agent** from the state, so 
        # we can route to the correct agent after collecting input.
        # For example, add a field to the state or use the last active agent.
        # or fill in `name` attribute of AI messages generated by the agents.
        active_agent = ... 

        return Command(
            update={
                "messages": [{
                    "role": "human",
                    "content": user_input,
                }]
            },
            goto=active_agent,
        )
    ```

See [how to implement multi-turn conversations](../how-tos/multi-agent-multi-turn-convo.ipynb) for a more detailed example.

#### Validating human input

If you need to validate the input provided by the human within the graph itself (rather than on the client side), you can achieve this by using multiple interrupt calls within a single node.

```python
from langgraph.types import interrupt

def human_node(state: State):
    """Human node with validation."""
    question = "What is your age?"

    while True:
        answer = interrupt(question)

        # Validate answer, if the answer isn't valid ask for input again.
        if not isinstance(answer, int) or answer < 0:
            question = f"'{answer} is not a valid age. What is your age?"
            answer = None
            continue
        else:
            # If the answer is valid, we can proceed.
            break
            
    print(f"The human in the loop is {answer} years old.")
    return {
        "age": answer
    }
```

### The `Command` primitive

When using the `interrupt` function, the graph will pause at the interrupt and wait for user input.

Graph execution can be resumed using the [Command](#langgraph.types.Command) primitive which can be passed through the `invoke`, `ainvoke`, `stream` or `astream` methods.

The `Command` primitive provides several options to control and modify the graph's state during resumption:

1. **Pass a value to the `interrupt`**: Provide data, such as a user's response, to the graph using `Command(resume=value)`. Execution resumes from the beginning of the node where the `interrupt` was used, however, this time the `interrupt(...)` call will return the value passed in the `Command(resume=value)` instead of pausing the graph.

       ```python
       # Resume graph execution with the user's input.
       graph.invoke(Command(resume={"age": "25"}), thread_config)
       ```

2. **Update the graph state**: Modify the graph state using `Command(update=update)`. Note that resumption starts from the beginning of the node where the `interrupt` was used. Execution resumes from the beginning of the node where the `interrupt` was used, but with the updated state.

      ```python
      # Update the graph state and resume.
      # You must provide a `resume` value if using an `interrupt`.
      graph.invoke(Command(update={"foo": "bar"}, resume="Let's go!!!"), thread_config)
      ```

By leveraging `Command`, you can resume graph execution, handle user inputs, and dynamically adjust the graph's state.

### Using with `invoke` and `ainvoke`

When you use `stream` or `astream` to run the graph, you will receive an `Interrupt` event that let you know the `interrupt` was triggered. 

`invoke` and `ainvoke` do not return the interrupt information. To access this information, you must use the [get_state](#langgraph.graph.graph.CompiledGraph.get_state) method to retrieve the graph state after calling `invoke` or `ainvoke`.

```python
## Run the graph up to the interrupt 
result = graph.invoke(inputs, thread_config)
## Get the graph state to get interrupt information.
state = graph.get_state(thread_config)
## Print the state values
print(state.values)
## Print the pending tasks
print(state.tasks)
## Resume the graph with the user's input.
graph.invoke(Command(resume={"age": "25"}), thread_config)
```

```pycon
{'foo': 'bar'} # State values
(
    PregelTask(
        id='5d8ffc92-8011-0c9b-8b59-9d3545b7e553', 
        name='node_foo', 
        path=('__pregel_pull', 'node_foo'), 
        error=None, 
        interrupts=(Interrupt(value='value_in_interrupt', resumable=True, ns=['node_foo:5d8ffc92-8011-0c9b-8b59-9d3545b7e553'], when='during'),), state=None, 
        result=None
    ),
) # Pending tasks. interrupts 
```

### How does resuming from an interrupt work?

!!! warning

    Resuming from an `interrupt` is **different** from Python's `input()` function, where execution resumes from the exact point where the `input()` function was called.

A critical aspect of using `interrupt` is understanding how resuming works. When you resume execution after an `interrupt`, graph execution starts from the **beginning** of the **graph node** where the last `interrupt` was triggered.

**All** code from the beginning of the node to the `interrupt` will be re-executed.

```python
counter = 0
def node(state: State):
    # All the code from the beginning of the node to the interrupt will be re-executed
    # when the graph resumes.
    global counter
    counter += 1
    print(f"> Entered the node: {counter} # of times")
    # Pause the graph and wait for user input.
    answer = interrupt()
    print("The value of counter is:", counter)
    ...
```

Upon **resuming** the graph, the counter will be incremented a second time, resulting in the following output:

```pycon
> Entered the node: 2 # of times
The value of counter is: 2
```

### Common Pitfalls

#### Side-effects

Place code with side effects, such as API calls, **after** the `interrupt` to avoid duplication, as these are re-triggered every time the node is resumed. 

=== "Side effects before interrupt (BAD)"

    This code will re-execute the API call another time when the node is resumed from
    the `interrupt`.

    This can be problematic if the API call is not idempotent or is just expensive.

    ```python
    from langgraph.types import interrupt

    def human_node(state: State):
        """Human node with validation."""
        api_call(...) # This code will be re-executed when the node is resumed.
        answer = interrupt(question)
    ```

=== "Side effects after interrupt (OK)"

    ```python
    from langgraph.types import interrupt

    def human_node(state: State):
        """Human node with validation."""
        
        answer = interrupt(question)
        
        api_call(answer) # OK as it's after the interrupt
    ```

=== "Side effects in a separate node (OK)"

    ```python
    from langgraph.types import interrupt

    def human_node(state: State):
        """Human node with validation."""
        
        answer = interrupt(question)
        
        return {
            "answer": answer
        }

    def api_call_node(state: State):
        api_call(...) # OK as it's in a separate node
    ```

#### Subgraphs called as functions

When invoking a subgraph [as a function](#as-a-function), the **parent graph** will resume execution from the **beginning of the node** where the subgraph was invoked (and where an `interrupt` was triggered). Similarly, the **subgraph**, will resume from the **beginning of the node** where the `interrupt()` function was called.

For example,

```python
def node_in_parent_graph(state: State):
    some_code()  # <-- This will re-execute when the subgraph is resumed.
    # Invoke a subgraph as a function.
    # The subgraph contains an `interrupt` call.
    subgraph_result = subgraph.invoke(some_input)
    ...
```

??? "**Example: Parent and Subgraph Execution Flow**"

      Say we have a parent graph with 3 nodes:

      **Parent Graph**: `node_1` ÔåÆ `node_2` (subgraph call) ÔåÆ `node_3`

      And the subgraph has 3 nodes, where the second node contains an `interrupt`:

      **Subgraph**: `sub_node_1` ÔåÆ `sub_node_2` (`interrupt`) ÔåÆ `sub_node_3`

      When resuming the graph, the execution will proceed as follows:

      1. **Skip `node_1`** in the parent graph (already executed, graph state was saved in snapshot).
      2. **Re-execute `node_2`** in the parent graph from the start.
      3. **Skip `sub_node_1`** in the subgraph (already executed, graph state was saved in snapshot).
      4. **Re-execute `sub_node_2`** in the subgraph from the beginning.
      5. Continue with `sub_node_3` and subsequent nodes.

      Here is abbreviated example code that you can use to understand how subgraphs work with interrupts.
      It counts the number of times each node is entered and prints the count.

      ```python
      import uuid
      from typing import TypedDict

      from langgraph.graph import StateGraph
      from langgraph.constants import START
      from langgraph.types import interrupt, Command
      from langgraph.checkpoint.memory import MemorySaver


      class State(TypedDict):
         """The graph state."""
         state_counter: int


      counter_node_in_subgraph = 0

      def node_in_subgraph(state: State):
         """A node in the sub-graph."""
         global counter_node_in_subgraph
         counter_node_in_subgraph += 1  # This code will **NOT** run again!
         print(f"Entered `node_in_subgraph` a total of {counter_node_in_subgraph} times")

      counter_human_node = 0

      def human_node(state: State):
         global counter_human_node
         counter_human_node += 1 # This code will run again!
         print(f"Entered human_node in sub-graph a total of {counter_human_node} times")
         answer = interrupt("what is your name?")
         print(f"Got an answer of {answer}")


      checkpointer = MemorySaver()

      subgraph_builder = StateGraph(State)
      subgraph_builder.add_node("some_node", node_in_subgraph)
      subgraph_builder.add_node("human_node", human_node)
      subgraph_builder.add_edge(START, "some_node")
      subgraph_builder.add_edge("some_node", "human_node")
      subgraph = subgraph_builder.compile(checkpointer=checkpointer)


      counter_parent_node = 0

      def parent_node(state: State):
         """This parent node will invoke the subgraph."""
         global counter_parent_node

         counter_parent_node += 1 # This code will run again on resuming!
         print(f"Entered `parent_node` a total of {counter_parent_node} times")
  
         # Please note that we're intentionally incrementing the state counter
         # in the graph state as well to demonstrate that the subgraph update
         # of the same key will not conflict with the parent graph (until
         subgraph_state = subgraph.invoke(state)
         return subgraph_state


      builder = StateGraph(State)
      builder.add_node("parent_node", parent_node)
      builder.add_edge(START, "parent_node")

      # A checkpointer must be enabled for interrupts to work!
      checkpointer = MemorySaver()
      graph = builder.compile(checkpointer=checkpointer)

      config = {
         "configurable": {
            "thread_id": uuid.uuid4(),
         }
      }

      for chunk in graph.stream({"state_counter": 1}, config):
         print(chunk)

      print('--- Resuming ---')

      for chunk in graph.stream(Command(resume="35"), config):
         print(chunk)
      ```

      This will print out

      ```pycon
      --- First invocation ---
      In parent node: {'foo': 'bar'}
      Entered `parent_node` a total of 1 times
      Entered `node_in_subgraph` a total of 1 times
      Entered human_node in sub-graph a total of 1 times
      {'__interrupt__': (Interrupt(value='what is your name?', resumable=True, ns=['parent_node:0b23d72f-aaba-0329-1a59-ca4f3c8bad3b', 'human_node:25df717c-cb80-57b0-7410-44e20aac8f3c'], when='during'),)}

      --- Resuming ---
      In parent node: {'foo': 'bar'}
      Entered `parent_node` a total of 2 times
      Entered human_node in sub-graph a total of 2 times
      Got an answer of 35
      {'parent_node': None} 
      ```



#### Using multiple interrupts

Using multiple interrupts within a **single** node can be helpful for patterns like [validating human input](#validating-human-input). However, using multiple interrupts in the same node can lead to unexpected behavior if not handled carefully.

When a node contains multiple interrupt calls, LangGraph keeps a list of resume values specific to the task executing the node. Whenever execution resumes, it starts at the beginning of the node. For each interrupt encountered, LangGraph checks if a matching value exists in the task's resume list. Matching is **strictly index-based**, so the order of interrupt calls within the node is critical.

To avoid issues, refrain from dynamically changing the node's structure between executions. This includes adding, removing, or reordering interrupt calls, as such changes can result in mismatched indices. These problems often arise from unconventional patterns, such as mutating state via `Command(resume=..., update=SOME_STATE_MUTATION)` or relying on global variables to modify the nodeÔÇÖs structure dynamically.

??? "Example of incorrect code"

    ```python
    import uuid
    from typing import TypedDict, Optional

    from langgraph.graph import StateGraph
    from langgraph.constants import START 
    from langgraph.types import interrupt, Command
    from langgraph.checkpoint.memory import MemorySaver


    class State(TypedDict):
        """The graph state."""

        age: Optional[str]
        name: Optional[str]


    def human_node(state: State):
        if not state.get('name'):
            name = interrupt("what is your name?")
        else:
            name = "N/A"

        if not state.get('age'):
            age = interrupt("what is your age?")
        else:
            age = "N/A"
            
        print(f"Name: {name}. Age: {age}")
        
        return {
            "age": age,
            "name": name,
        }


    builder = StateGraph(State)
    builder.add_node("human_node", human_node)
    builder.add_edge(START, "human_node")

    # A checkpointer must be enabled for interrupts to work!
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
        }
    }

    for chunk in graph.stream({"age": None, "name": None}, config):
        print(chunk)

    for chunk in graph.stream(Command(resume="John", update={"name": "foo"}), config):
        print(chunk)
    ```

    ```pycon
    {'__interrupt__': (Interrupt(value='what is your name?', resumable=True, ns=['human_node:3a007ef9-c30d-c357-1ec1-86a1a70d8fba'], when='during'),)}
    Name: N/A. Age: John
    {'human_node': {'age': 'John', 'name': 'N/A'}}
    ```

### Additional Resources ­ƒôÜ

- [**Conceptual Guide: Persistence**](#replay): Read the persistence guide for more context on replaying.
- [**How to Guides: Human-in-the-loop**](#human-in-the-loop): Learn how to implement human-in-the-loop workflows in LangGraph.
- [**How to implement multi-turn conversations**](../how-tos/multi-agent-multi-turn-convo.ipynb): Learn how to implement multi-turn conversations in LangGraph.


<a name="docsconceptsindexmd"></a>

## Conceptual Guide

This guide provides explanations of the key concepts behind the LangGraph framework and AI applications more broadly.

We recommend that you go through at least the [Quick Start](../tutorials/introduction.ipynb) before diving into the conceptual guide. This will provide practical context that will make it easier to understand the concepts discussed here.

The conceptual guide does not cover step-by-step instructions or specific implementation examples ÔÇö those are found in the [Tutorials](#docstutorialsindexmd) and [How-to guides](#docshow-tosindexmd). For detailed reference material, please see the [API reference](#docsreferenceindexmd).

### LangGraph

**High Level**

- [Why LangGraph?](#docsconceptshigh_levelmd): A high-level overview of LangGraph and its goals.

**Concepts**

- [LangGraph Glossary](#docsconceptslow_levelmd): LangGraph workflows are designed as graphs, with nodes representing different components and edges representing the flow of information between them. This guide provides an overview of the key concepts associated with LangGraph graph primitives.
- [Common Agentic Patterns](#docsconceptsagentic_conceptsmd): An agent uses an LLM to pick its own control flow to solve more complex problems! Agents are a key building block in many LLM applications. This guide explains the different types of agent architectures and how they can be used to control the flow of an application.
- [Multi-Agent Systems](#docsconceptsmulti_agentmd): Complex LLM applications can often be broken down into multiple agents, each responsible for a different part of the application. This guide explains common patterns for building multi-agent systems.
- [Breakpoints](#docsconceptsbreakpointsmd): Breakpoints allow pausing the execution of a graph at specific points. Breakpoints allow stepping through graph execution for debugging purposes.
- [Human-in-the-Loop](#docsconceptshuman_in_the_loopmd): Explains different ways of integrating human feedback into a LangGraph application.
- [Time Travel](#docsconceptstime-travelmd): Time travel allows you to replay past actions in your LangGraph application to explore alternative paths and debug issues.
- [Persistence](#docsconceptspersistencemd): LangGraph has a built-in persistence layer, implemented through checkpointers. This persistence layer helps to support powerful capabilities like human-in-the-loop, memory, time travel, and fault-tolerance.
- [Memory](#docsconceptsmemorymd): Memory in AI applications refers to the ability to process, store, and effectively recall information from past interactions. With memory, your agents can learn from feedback and adapt to users' preferences.  
- [Streaming](#docsconceptsstreamingmd): Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs. 
- [FAQ](#docsconceptsfaqmd): Frequently asked questions about LangGraph.

### LangGraph Platform

LangGraph Platform is a commercial solution for deploying agentic applications in production, built on the open-source LangGraph framework.

The LangGraph Platform offers a few different deployment options described in the [deployment options guide](#docsconceptsdeployment_optionsmd).


!!! tip

    * LangGraph is an MIT-licensed open-source library, which we are committed to maintaining and growing for the community.
    * You can always deploy LangGraph applications on your own infrastructure using the open-source LangGraph project without using LangGraph Platform.

#### High Level

- [Why LangGraph Platform?](#docsconceptslanggraph_platformmd): The LangGraph platform is an opinionated way to deploy and manage LangGraph applications. This guide provides an overview of the key features and concepts behind LangGraph Platform.
- [Deployment Options](#docsconceptsdeployment_optionsmd): LangGraph Platform offers four deployment options: [Self-Hosted Lite](#self-hosted-lite), [Self-Hosted Enterprise](#self-hosted-enterprise), [bring your own cloud (BYOC)](#docsconceptsbring_your_own_cloudmd), and [Cloud SaaS](#docsconceptslanggraph_cloudmd). This guide explains the differences between these options, and which Plans they are available on.
- [Plans](#docsconceptsplansmd): LangGraph Platforms offer three different plans: Developer, Plus, Enterprise. This guide explains the differences between these options, what deployment options are available for each, and how to sign up for each one.
- [Template Applications](#docsconceptstemplate_applicationsmd): Reference applications designed to help you get started quickly when building with LangGraph.

#### Components

The LangGraph Platform comprises several components that work together to support the deployment and management of LangGraph applications:

- [LangGraph Server](#docsconceptslanggraph_servermd): The LangGraph Server is designed to support a wide range of agentic application use cases, from background processing to real-time interactions. 
- [LangGraph Studio](#docsconceptslanggraph_studiomd): LangGraph Studio is a specialized IDE that can connect to a LangGraph Server to enable visualization, interaction, and debugging of the application locally.
- [LangGraph CLI](#docsconceptslanggraph_climd): LangGraph CLI is a command-line interface that helps to interact with a local LangGraph
- [Python/JS SDK](#docsconceptssdkmd): The Python/JS SDK provides a programmatic way to interact with deployed LangGraph Applications.
- [Remote Graph](#docshow-tosuse-remote-graphmd): A RemoteGraph allows you to interact with any deployed LangGraph application as though it were running locally.

#### LangGraph Server

- [Application Structure](#docsconceptsapplication_structuremd): A LangGraph application consists of one or more graphs, a LangGraph API Configuration file (`langgraph.json`), a file that specifies dependencies, and environment variables.
- [Assistants](#docsconceptsassistantsmd): Assistants are a way to save and manage different configurations of your LangGraph applications.
- [Web-hooks](#webhooks): Webhooks allow your running LangGraph application to send data to external services on specific events.
- [Cron Jobs](#cron-jobs): Cron jobs are a way to schedule tasks to run at specific times in your LangGraph application.
- [Double Texting](#docsconceptsdouble_textingmd): Double texting is a common issue in LLM applications where users may send multiple messages before the graph has finished running. This guide explains how to handle double texting with LangGraph Deploy.
- [Authentication & Access Control](#docsconceptsauthmd): Learn about options for authentication and access control when deploying the LangGraph Platform.

#### Deployment Options


- [Self-Hosted Lite](#docsconceptsself_hostedmd): A free (up to 1 million nodes executed), limited version of LangGraph Platform that you can run locally or in a self-hosted manner
- [Cloud SaaS](#docsconceptslanggraph_cloudmd): Hosted as part of LangSmith.
- [Bring Your Own Cloud](#docsconceptsbring_your_own_cloudmd): We manage the infrastructure, so you don't have to, but the infrastructure all runs within your cloud.
- [Self-Hosted Enterprise](#docsconceptsself_hostedmd): Completely managed by you.

<a name="docsconceptslanggraph_climd"></a>

## LangGraph CLI

!!! info "Prerequisites"
    - [LangGraph Platform](#docsconceptslanggraph_platformmd)
    - [LangGraph Server](#docsconceptslanggraph_servermd)

The LangGraph CLI is a multi-platform command-line tool for building and running the [LangGraph API server](#docsconceptslanggraph_servermd) locally. This offers an alternative to the [LangGraph Studio desktop app](#docsconceptslanggraph_studiomd) for developing and testing agents across all major operating systems (Linux, Windows, MacOS). The resulting server includes all API endpoints for your graph's runs, threads, assistants, etc. as well as the other services required to run your agent, including a managed database for checkpointing and storage.

### Installation

The LangGraph CLI can be installed via Homebrew (on macOS) or pip:

=== "Homebrew"
    ```bash
    brew install langgraph-cli
    ```

=== "pip" 
    ```bash
    pip install langgraph-cli
    ```

### Commands

The CLI provides the following core functionality:

#### `build`

The `langgraph build` command builds a Docker image for the [LangGraph API server](#docsconceptslanggraph_servermd) that can be directly deployed.

#### `dev`

!!! note "New in version 0.1.55"
    The `langgraph dev` command was introduced in langgraph-cli version 0.1.55.

!!! note "Python only"

    Currently, the CLI only supports Python >= 3.11.
    JS support is coming soon.

The `langgraph dev` command starts a lightweight development server that requires no Docker installation. This server is ideal for rapid development and testing, with features like:

- Hot reloading: Changes to your code are automatically detected and reloaded
- Debugger support: Attach your IDE's debugger for line-by-line debugging
- In-memory state with local persistence: Server state is stored in memory for speed but persisted locally between restarts

To use this command, you need to install the CLI with the "inmem" extra:

```bash
pip install -U "langgraph-cli[inmem]"
```

**Note**: This command is intended for local development and testing only. It is not recommended for production use. Since it does not use Docker, we recommend using virtual environments to manage your project's dependencies.

#### `up`

The `langgraph up` command starts an instance of the [LangGraph API server](#docsconceptslanggraph_servermd) locally in a docker container. This requires thedocker server to be running locally. It also requires a LangSmith API key for local development or a license key for production use.

The server includes all API endpoints for your graph's runs, threads, assistants, etc. as well as the other services required to run your agent, including a managed database for checkpointing and storage.

#### `dockerfile`

The `langgraph dockerfile` command generates a [Dockerfile](https://docs.docker.com/reference/dockerfile/) that can be used to build images for and deploy instances of the [LangGraph API server](#docsconceptslanggraph_servermd). This is useful if you want to further customize the dockerfile or deploy in a more custom way.

### Related

- [LangGraph CLI API Reference](#docscloudreferenceclimd)


<a name="docsconceptslanggraph_cloudmd"></a>

## Cloud SaaS

!!! info "Prerequisites" 
    - [LangGraph Platform](#docsconceptslanggraph_platformmd) 
    - [LangGraph Server](#docsconceptslanggraph_servermd)

### Overview

LangGraph's Cloud SaaS is a managed service for deploying LangGraph APIs, regardless of its definition or dependencies. The service offers managed implementations of checkpointers and stores, allowing you to focus on building the right cognitive architecture for your use case. By handling scalable & secure infrastructure, LangGraph Cloud offers the fastest path to getting your LangGraph API deployed to production.

### Deployment

A **deployment** is an instance of a LangGraph API. A single deployment can have many [revisions](#revision). When a deployment is created, all the necessary infrastructure (e.g. database, containers, secrets store) are automatically provisioned. See the [architecture diagram](#architecture) below for more details.

See the [how-to guide](#create-new-deployment) for creating a new deployment.

### Resource Allocation

| **Deployment Type** | **CPU** | **Memory** | **Scaling**         |
|---------------------|---------|------------|---------------------|
| Development         | 1 CPU   | 1 GB       | Up to 1 container   |
| Production          | 2 CPU   | 2 GB       | Up to 10 containers |

### Autoscaling
`Production` type deployments automatically scale up to 10 containers. Scaling is based on the current request load for a single container. Specifically, the autoscaling implementation scales the deployment so that each container is processing about 10 concurrent requests. For example... 

- If the deployment is processing 20 concurrent requests, the deployment will scale up from 1 container to 2 containers (20 requests / 2 containers = 10 requests per container).
- If a deployment of 2 containers is processing 10 requests, the deployment will scale down from 2 containers to 1 container (10 requests / 1 container = 10 requests per container).

10 concurrent requests per container is the target threshold. However, 10 concurrent requests per container is not a hard limit. The number of concurrent requests can exceed 10 if there is a sudden burst of requests.

Scale down actions are delayed for 30 minutes before any action is taken. In other words, if the autoscaling implementation decides to scale down a deployment, it will first wait for 30 minutes before scaling down. After 30 minutes, the concurrency metric is recomputed and the deployment will scale down if the concurrency metric has met the target threshold. Otherwise, the deployment remains scaled up. This "cool down" period ensures that deployments do not scale up and down too frequently.

In the future, the autoscaling implementation may evolve to accommodate other metrics such as background run queue size.

### Revision

A revision is an iteration of a [deployment](#deployment). When a new deployment is created, an initial revision is automatically created. To deploy new code changes or update environment variable configurations for a deployment, a new revision must be created. When a revision is created, a new container image is built automatically.

See the [how-to guide](#create-new-revision) for creating a new revision.

### Asynchronous Deployment

Infrastructure for [deployments](#deployment) and [revisions](#revision) are provisioned and deployed asynchronously. They are not deployed immediately after submission. Currently, deployment can take up to several minutes.

- When a new deployment is created, a new database is created for the deployment. Database creation is a one-time step. This step contributes to a longer deployment time for the initial revision of the deployment.
- When a subsequent revision is created for a deployment, there is no database creation step. The deployment time for a subsequent revision is significantly faster compared to the deployment time of the initial revision.
- The deployment process for each revision contains a build step, which can take up to a few minutes.

!!! info "Database creation for `Development` type deployments takes longer than database creation for `Production` type deployments."

### Architecture

!!! warning "Subject to Change"
The Cloud SaaS deployment architecture may change in the future.

A high-level diagram of a Cloud SaaS deployment.

![diagram](img/langgraph_cloud_architecture.png)

### Related

- [Deployment Options](#docsconceptsdeployment_optionsmd)


<a name="docsconceptslanggraph_platformmd"></a>

## LangGraph Platform

### Overview

LangGraph Platform is a commercial solution for deploying agentic applications to production, built on the open-source [LangGraph framework](#docsconceptshigh_levelmd).

The LangGraph Platform consists of several components that work together to support the development, deployment, debugging, and monitoring of LangGraph applications:

- [LangGraph Server](#docsconceptslanggraph_servermd): The server defines an opinionated API and architecture that incorporates best practices for deploying agentic applications, allowing you to focus on building your agent logic rather than developing server infrastructure.
- [LangGraph Studio](#docsconceptslanggraph_studiomd): LangGraph Studio is a specialized IDE that can connect to a LangGraph Server to enable visualization, interaction, and debugging of the application locally.
- [LangGraph CLI](#docsconceptslanggraph_climd): LangGraph CLI is a command-line interface that helps to interact with a local LangGraph
- [Python/JS SDK](#docsconceptssdkmd): The Python/JS SDK provides a programmatic way to interact with deployed LangGraph Applications.
- [Remote Graph](#docshow-tosuse-remote-graphmd): A RemoteGraph allows you to interact with any deployed LangGraph application as though it were running locally.

![](img/lg_platform.png)

The LangGraph Platform offers a few different deployment options described in the [deployment options guide](#docsconceptsdeployment_optionsmd).

### Why Use LangGraph Platform?

**LangGraph Platform** handles common issues that arise when deploying LLM applications to production, allowing you to focus on agent logic instead of managing server infrastructure.

- **[Streaming Support](#docsconceptsstreamingmd)**: As agents grow more sophisticated, they often benefit from streaming both token outputs and intermediate states back to the user. Without this, users are left waiting for potentially long operations with no feedback. LangGraph Server provides [multiple streaming modes](#docsconceptsstreamingmd) optimized for various application needs.

- **Background Runs**: For agents that take longer to process (e.g., hours), maintaining an open connection can be impractical. The LangGraph Server supports launching agent runs in the background and provides both polling endpoints and webhooks to monitor run status effectively.
 
- **Support for long runs**: Vanilla server setups often encounter timeouts or disruptions when handling requests that take a long time to complete. LangGraph ServerÔÇÖs API provides robust support for these tasks by sending regular heartbeat signals, preventing unexpected connection closures during prolonged processes.

- **Handling Burstiness**: Certain applications, especially those with real-time user interaction, may experience "bursty" request loads where numerous requests hit the server simultaneously. LangGraph Server includes a task queue, ensuring requests are handled consistently without loss, even under heavy loads.

- **[Double Texting](#docsconceptsdouble_textingmd)**: In user-driven applications, itÔÇÖs common for users to send multiple messages rapidly. This ÔÇ£double textingÔÇØ can disrupt agent flows if not handled properly. LangGraph Server offers built-in strategies to address and manage such interactions.

- **[Checkpointers and Memory Management](#checkpoints)**: For agents needing persistence (e.g., conversation memory), deploying a robust storage solution can be complex. LangGraph Platform includes optimized [checkpointers](#checkpoints) and a [memory store](#memory-store), managing state across sessions without the need for custom solutions.

- **[Human-in-the-loop Support](#docsconceptshuman_in_the_loopmd)**: In many applications, users require a way to intervene in agent processes. LangGraph Server provides specialized endpoints for human-in-the-loop scenarios, simplifying the integration of manual oversight into agent workflows.

By using LangGraph Platform, you gain access to a robust, scalable deployment solution that mitigates these challenges, saving you the effort of implementing and maintaining them manually. This allows you to focus more on building effective agent behavior and less on solving deployment infrastructure issues.


<a name="docsconceptslanggraph_servermd"></a>

## LangGraph Server

!!! info "Prerequisites"
    - [LangGraph Platform](#docsconceptslanggraph_platformmd)
    - [LangGraph Glossary](#docsconceptslow_levelmd)

### Overview

LangGraph Server offers an API for creating and managing agent-based applications. It is built on the concept of [assistants](#docsconceptsassistantsmd), which are agents configured for specific tasks, and includes built-in [persistence](#memory-store) and a **task queue**. This versatile API supports a wide range of agentic application use cases, from background processing to real-time interactions.

### Key Features

The LangGraph Platform incorporates best practices for agent deployment, so you can focus on building your agent logic.

* **Streaming endpoints**: Endpoints that expose [multiple different streaming modes](#docsconceptsstreamingmd). We've made these work even for long-running agents that may go minutes between consecutive stream events.
* **Background runs**: The LangGraph Server supports launching assistants in the background with endpoints for polling the status of the assistant's run and webhooks to monitor run status effectively.
- **Support for long runs**: Our blocking endpoints for running assistants send regular heartbeat signals, preventing unexpected connection closures when handling requests that take a long time to complete.
* **Task queue**: We've added a task queue to make sure we don't drop any requests if they arrive in a bursty nature.
* **Horizontally scalable infrastructure**: LangGraph Server is designed to be horizontally scalable, allowing you to scale up and down your usage as needed.
* **Double texting support**: Many times users might interact with your graph in unintended ways. For instance, a user may send one message and before the graph has finished running send a second message. We call this ["double texting"](#docsconceptsdouble_textingmd) and have added four different ways to handle this.
* **Optimized checkpointer**: LangGraph Platform comes with a built-in [checkpointer](#checkpoints) optimized for LangGraph applications.
* **Human-in-the-loop endpoints**: We've exposed all endpoints needed to support [human-in-the-loop](#docsconceptshuman_in_the_loopmd) features.
* **Memory**: In addition to thread-level persistence (covered above by [checkpointers]l(./persistence.md#checkpoints)), LangGraph Platform also comes with a built-in [memory store](#memory-store).
* **Cron jobs**: Built-in support for scheduling tasks, enabling you to automate regular actions like data clean-up or batch processing within your applications.
* **Webhooks**: Allows your application to send real-time notifications and data updates to external systems, making it easy to integrate with third-party services and trigger actions based on specific events.
* **Monitoring**: LangGraph Server integrates seamlessly with the [LangSmith](https://docs.smith.langchain.com/) monitoring platform, providing real-time insights into your application's performance and health.

### What are you deploying?

When you deploy a LangGraph Server, you are deploying one or more [graphs](#graphs), a database for [persistence](#docsconceptspersistencemd), and a task queue.

#### Graphs

When you deploy a graph with LangGraph Server, you are deploying a "blueprint" for an [Assistant](#docsconceptsassistantsmd). 

An [Assistant](#docsconceptsassistantsmd) is a graph paired with specific configuration settings. You can create multiple assistants per graph, each with unique settings to accommodate different use cases
that can be served by the same graph.

Upon deployment, LangGraph Server will automatically create a default assistant for each graph using the graph's default configuration settings.

You can interact with assistants through the [LangGraph Server API](#langgraph-server-api).

!!! note

    We often think of a graph as implementing an [agent](#docsconceptsagentic_conceptsmd), but a graph does not necessarily need to implement an agent. For example, a graph could implement a simple
    chatbot that only supports back-and-forth conversation, without the ability to influence any application control flow. In reality, as applications get more complex, a graph will often implement a more complex flow that may use [multiple agents](#docsconceptsmulti_agentmd) working in tandem.

#### Persistence and Task Queue

The LangGraph Server leverages a database for [persistence](#docsconceptspersistencemd) and a task queue.

Currently, only [Postgres](https://www.postgresql.org/) is supported as a database for LangGraph Server and [Redis](https://redis.io/) as the task queue.

If you're deploying using [LangGraph Cloud](#docsconceptslanggraph_cloudmd), these components are managed for you. If you're deploying LangGraph Server on your own infrastructure, you'll need to set up and manage these components yourself.

Please review the [deployment options](#docsconceptsdeployment_optionsmd) guide for more information on how these components are set up and managed.

### Application Structure

To deploy a LangGraph Server application, you need to specify the graph(s) you want to deploy, as well as any relevant configuration settings, such as dependencies and environment variables.

Read the [application structure](#docsconceptsapplication_structuremd) guide to learn how to structure your LangGraph application for deployment.

### LangGraph Server API

The LangGraph Server API allows you to create and manage [assistants](#docsconceptsassistantsmd), [threads](#threads), [runs](#runs), [cron jobs](#cron-jobs), and more.

The [LangGraph Cloud API Reference](../cloud/reference/api/api_ref.html) provides detailed information on the API endpoints and data models.

#### Assistants

An [Assistant](#docsconceptsassistantsmd) refers to a [graph](#graphs) plus specific [configuration](#configuration) settings for that graph.

You can think of an assistant as a saved configuration of an [agent](#docsconceptsagentic_conceptsmd).

When building agents, it is fairly common to make rapid changes that *do not* alter the graph logic. For example, simply changing prompts or the LLM selection can have significant impacts on the behavior of the agents. Assistants offer an easy way to make and save these types of changes to agent configuration.

#### Threads

A thread contains the accumulated state of a sequence of [runs](#runs). If a run is executed on a thread, then the [state](#state) of the underlying graph of the assistant will be persisted to the thread.

A thread's current and historical state can be retrieved. To persist state, a thread must be created prior to executing a run.

The state of a thread at a particular point in time is called a [checkpoint](#checkpoints). Checkpoints can be used to restore the state of a thread at a later time.

For more on threads and checkpoints, see this section of the [LangGraph conceptual guide](#persistence).

The LangGraph Cloud API provides several endpoints for creating and managing threads and thread state. See the [API reference](#tag\threads) for more details.

#### Runs

A run is an invocation of an [assistant](#assistants). Each run may have its own input, configuration, and metadata, which may affect execution and output of the underlying graph. A run can optionally be executed on a [thread](#threads).

The LangGraph Cloud API provides several endpoints for creating and managing runs. See the [API reference](#tag\thread-runs\) for more details.

#### Store

Store is an API for managing persistent [key-value store](#memory-store) that is available from any [thread](#threads).

Stores are useful for implementing [memory](#docsconceptsmemorymd) in your LangGraph application.

#### Cron Jobs

There are many situations in which it is useful to run an assistant on a schedule. 

For example, say that you're building an assistant that runs daily and sends an email summary
of the day's news. You could use a cron job to run the assistant every day at 8:00 PM.

LangGraph Cloud supports cron jobs, which run on a user-defined schedule. The user specifies a schedule, an assistant, and some input. After that, on the specified schedule, the server will:

- Create a new thread with the specified assistant
- Send the specified input to that thread

Note that this sends the same input to the thread every time. See the [how-to guide](#docscloudhow-toscron_jobsmd) for creating cron jobs.

The LangGraph Cloud API provides several endpoints for creating and managing cron jobs. See the [API reference](#tag\runscreate\POST\threads\{thread_id}\runs\crons) for more details.

#### Webhooks

Webhooks enable event-driven communication from your LangGraph Cloud application to external services. For example, you may want to issue an update to a separate service once an API call to LangGraph Cloud has finished running.

Many LangGraph Cloud endpoints accept a `webhook` parameter. If this parameter is specified by a an endpoint that can accept POST requests, LangGraph Cloud will send a request at the completion of a run.

See the corresponding [how-to guide](#docscloudhow-toswebhooksmd) for more detail.

### Related

* LangGraph [Application Structure](#docsconceptsapplication_structuremd) guide explains how to structure your LangGraph application for deployment.
* [How-to guides for the LangGraph Platform](#docshow-tosindexmd).
* The [LangGraph Cloud API Reference](../cloud/reference/api/api_ref.html) provides detailed information on the API endpoints and data models.


<a name="docsconceptslanggraph_studiomd"></a>

## LangGraph Studio

!!! info "Prerequisites"

    - [LangGraph Platform](#docsconceptslanggraph_platformmd)
    - [LangGraph Server](#docsconceptslanggraph_servermd)

LangGraph Studio offers a new way to develop LLM applications by providing a specialized agent IDE that enables visualization, interaction, and debugging of complex agentic applications.

With visual graphs and the ability to edit state, you can better understand agent workflows and iterate faster. LangGraph Studio integrates with LangSmith allowing you to  collaborate with teammates to debug failure modes.

![](img/lg_studio.png)

### Features

The key features of LangGraph Studio are:

- Visualizes your graph
- Test your graph by running it from the UI
- Debug your agent by [modifying its state and rerunning](#docsconceptshuman_in_the_loopmd)
- Create and manage [assistants](#docsconceptsassistantsmd)
- View and manage [threads](#threads)
- View and manage [long term memory](#docsconceptsmemorymd)
- Add node input/outputs to [LangSmith](https://smith.langchain.com/) datasets for testing

### Types

#### Development server with web UI

You can [run a local in-memory development server](#docstutorialslanggraph-platformlocal-servermd) that can be used to connect a local LangGraph app with a web version of the studio.
For example, if you start the local server with `langgraph dev` (running at `http://127.0.0.1:2024` by default), you can connect to the studio by navigating to:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

See [instructions here](#dev) for more information.

The web UI version of the studio will connect to your locally running server ÔÇö your agent is still running locally and never leaves your device.

#### Cloud studio

If you have deployed your LangGraph application on LangGraph Platform (Cloud), you can access the studio as part of that

#### Desktop app

LangGraph Studio is available as a [desktop app](https://studio.langchain.com/) for MacOS users.

While in Beta, LangGraph Studio is available for free to all [LangSmith](https://smith.langchain.com/) users on any plan tier.

### Studio FAQs

#### Why is my project failing to start?

There are a few reasons that your project might fail to start, here are some of the most common ones.

##### Docker issues (desktop only)

LangGraph Studio (desktop) requires Docker Desktop version 4.24 or higher. Please make sure you have a version of Docker installed that satisfies that requirement and also make sure you have the Docker Desktop app up and running before trying to use LangGraph Studio. In addition, make sure you have docker-compose updated to version 2.22.0 or higher.

##### Configuration or environment issues

Another reason your project might fail to start is because your configuration file is defined incorrectly, or you are missing required environment variables. 

#### How does interrupt work?

When you select the `Interrupts` dropdown and select a node to interrupt the graph will pause execution before and after (unless the node goes straight to `END`) that node has run. This means that you will be able to both edit the state before the node is ran and the state after the node has ran. This is intended to allow developers more fine-grained control over the behavior of a node and make it easier to observe how the node is behaving. You will not be able to edit the state after the node has ran if the node is the final node in the graph.

#### How do I reload the app?  (desktop only)

If you would like to reload the app, don't use Command+R as you might normally do. Instead, close and reopen the app for a full refresh.

#### How does automatic rebuilding work?  (desktop only)

One of the key features of LangGraph Studio is that it automatically rebuilds your image when you change the source code. This allows for a super fast development and testing cycle which makes it easy to iterate on your graph. There are two different ways that LangGraph rebuilds your image: either by editing the image or completely rebuilding it.

##### Rebuilds from source code changes

If you modified the source code only (no configuration or dependency changes!) then the image does not require a full rebuild, and LangGraph Studio will only update the relevant parts. The UI status in the bottom left will switch from `Online` to `Stopping` temporarily while the image gets edited. The logs will be shown as this process is happening, and after the image has been edited the status will change back to `Online` and you will be able to run your graph with the modified code!


##### Rebuilds from configuration or dependency changes

If you edit your graph configuration file (`langgraph.json`) or the dependencies (either `pyproject.toml` or `requirements.txt`) then the entire image will be rebuilt. This will cause the UI to switch away from the graph view and start showing the logs of the new image building process. This can take a minute or two, and once it is done your updated image will be ready to use!

#### Why is my graph taking so long to startup?  (desktop only)

The LangGraph Studio interacts with a local LangGraph API server. To stay aligned with ongoing updates, the LangGraph API requires regular rebuilding. As a result, you may occasionally experience slight delays when starting up your project.

### Why are extra edges showing up in my graph?

If you don't define your conditional edges carefully, you might notice extra edges appearing in your graph. This is because without proper definition, LangGraph Studio assumes the conditional edge could access all other nodes. In order for this to not be the case, you need to be explicit about how you define the nodes the conditional edge routes to. There are two ways you can do this:

#### Solution 1: Include a path map

The first way to solve this is to add path maps to your conditional edges. A path map is just a dictionary or array that maps the possible outputs of your router function with the names of the nodes that each output corresponds to. The path map is passed as the third argument to the `add_conditional_edges` function like so:

=== "Python"

    ```python
    graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
    ```

=== "Javascript"

    ```ts
    graph.addConditionalEdges("node_a", routingFunction, { true: "node_b", false: "node_c" });
    ```

In this case, the routing function returns either True or False, which map to `node_b` and `node_c` respectively.

#### Solution 2: Update the typing of the router (Python only)

Instead of passing a path map, you can also be explicit about the typing of your routing function by specifying the nodes it can map to using the `Literal` python definition. Here is an example of how to define a routing function in that way:

```python
def routing_function(state: GraphState) -> Literal["node_b","node_c"]:
    if state['some_condition'] == True:
        return "node_b"
    else:
        return "node_c"
```


### Related

For more information please see the following:

*  [LangGraph Studio how-to guides](#langgraph-studio)

<a name="docsconceptslow_levelmd"></a>

## LangGraph Glossary

### Graphs

At its core, LangGraph models agent workflows as graphs. You define the behavior of your agents using three key components:

1. [`State`](#state): A shared data structure that represents the current snapshot of your application. It can be any Python type, but is typically a `TypedDict` or Pydantic `BaseModel`.

2. [`Nodes`](#nodes): Python functions that encode the logic of your agents. They receive the current `State` as input, perform some computation or side-effect, and return an updated `State`.

3. [`Edges`](#edges): Python functions that determine which `Node` to execute next based on the current `State`. They can be conditional branches or fixed transitions.

By composing `Nodes` and `Edges`, you can create complex, looping workflows that evolve the `State` over time. The real power, though, comes from how LangGraph manages that `State`. To emphasize: `Nodes` and `Edges` are nothing more than Python functions - they can contain an LLM or just good ol' Python code.

In short: _nodes do the work. edges tell what to do next_.

LangGraph's underlying graph algorithm uses [message passing](https://en.wikipedia.org/wiki/Message_passing) to define a general program. When a Node completes its operation, it sends messages along one or more edges to other node(s). These recipient nodes then execute their functions, pass the resulting messages to the next set of nodes, and the process continues. Inspired by Google's [Pregel](https://research.google/pubs/pregel-a-system-for-large-scale-graph-processing/) system, the program proceeds in discrete "super-steps."

A super-step can be considered a single iteration over the graph nodes. Nodes that run in parallel are part of the same super-step, while nodes that run sequentially belong to separate super-steps. At the start of graph execution, all nodes begin in an `inactive` state. A node becomes `active` when it receives a new message (state) on any of its incoming edges (or "channels"). The active node then runs its function and responds with updates. At the end of each super-step, nodes with no incoming messages vote to `halt` by marking themselves as `inactive`. The graph execution terminates when all nodes are `inactive` and no messages are in transit.

#### StateGraph

The `StateGraph` class is the main graph class to use. This is parameterized by a user defined `State` object.

#### MessageGraph

The `MessageGraph` class is a special type of graph. The `State` of a `MessageGraph` is ONLY a list of messages. This class is rarely used except for chatbots, as most applications require the `State` to be more complex than a list of messages.

#### Compiling your graph

To build your graph, you first define the [state](#state), you then add [nodes](#nodes) and [edges](#edges), and then you compile it. What exactly is compiling your graph and why is it needed?

Compiling is a pretty simple step. It provides a few basic checks on the structure of your graph (no orphaned nodes, etc). It is also where you can specify runtime args like [checkpointers](#docsconceptspersistencemd) and [breakpoints](#breakpoints). You compile your graph by just calling the `.compile` method:

```python
graph = graph_builder.compile(...)
```

You **MUST** compile your graph before you can use it.

### State

The first thing you do when you define a graph is define the `State` of the graph. The `State` consists of the [schema of the graph](#schema) as well as [`reducer` functions](#reducers) which specify how to apply updates to the state. The schema of the `State` will be the input schema to all `Nodes` and `Edges` in the graph, and can be either a `TypedDict` or a `Pydantic` model. All `Nodes` will emit updates to the `State` which are then applied using the specified `reducer` function.

#### Schema

The main documented way to specify the schema of a graph is by using `TypedDict`. However, we also support [using a Pydantic BaseModel](../how-tos/state-model.ipynb) as your graph state to add **default values** and additional data validation.

By default, the graph will have the same input and output schemas. If you want to change this, you can also specify explicit input and output schemas directly. This is useful when you have a lot of keys, and some are explicitly for input and others for output. See the [notebook here](../how-tos/input_output_schema.ipynb) for how to use.

##### Multiple schemas

Typically, all graph nodes communicate with a single schema. This means that they will read and write to the same state channels. But, there are cases where we want more control over this:

- Internal nodes can pass information that is not required in the graph's input / output.
- We may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.

It is possible to have nodes write to private state channels inside the graph for internal node communication. We can simply define a private schema, `PrivateState`. See [this notebook](../how-tos/pass_private_state.ipynb) for more detail.

It is also possible to define explicit input and output schemas for a graph. In these cases, we define an "internal" schema that contains _all_ keys relevant to graph operations. But, we also define `input` and `output` schemas that are sub-sets of the "internal" schema to constrain the input and output of the graph. See [this notebook](../how-tos/input_output_schema.ipynb) for more detail.

Let's look at an example:

```python
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState,input=InputState,output=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})
{'graph_output': 'My name is Lance'}
```

There are two subtle and important points to note here:

1. We pass `state: InputState` as the input schema to `node_1`. But, we write out to `foo`, a channel in `OverallState`. How can we write out to a state channel that is not included in the input schema? This is because a node _can write to any state channel in the graph state._ The graph state is the union of of the state channels defined at initialization, which includes `OverallState` and the filters `InputState` and `OutputState`.

2. We initialize the graph with `StateGraph(OverallState,input=InputState,output=OutputState)`. So, how can we write to `PrivateState` in `node_2`? How does the graph gain access to this schema if it was not passed in the `StateGraph` initialization? We can do this because _nodes can also declare additional state channels_ as long as the state schema definition exists. In this case, the `PrivateState` schema is defined, so we can add `bar` as a new state channel in the graph and write to it.

#### Reducers

Reducers are key to understanding how updates from nodes are applied to the `State`. Each key in the `State` has its own independent reducer function. If no reducer function is explicitly specified then it is assumed that all updates to that key should override it. There are a few different types of reducers, starting with the default type of reducer:

##### Default Reducer

These two examples show how to use the default reducer:

**Example A:**

```python
from typing_extensions import TypedDict

class State(TypedDict):
    foo: int
    bar: list[str]
```

In this example, no reducer functions are specified for any key. Let's assume the input to the graph is `{"foo": 1, "bar": ["hi"]}`. Let's then assume the first `Node` returns `{"foo": 2}`. This is treated as an update to the state. Notice that the `Node` does not need to return the whole `State` schema - just an update. After applying this update, the `State` would then be `{"foo": 2, "bar": ["hi"]}`. If the second node returns `{"bar": ["bye"]}` then the `State` would then be `{"foo": 2, "bar": ["bye"]}`

**Example B:**

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

In this example, we've used the `Annotated` type to specify a reducer function (`operator.add`) for the second key (`bar`). Note that the first key remains unchanged. Let's assume the input to the graph is `{"foo": 1, "bar": ["hi"]}`. Let's then assume the first `Node` returns `{"foo": 2}`. This is treated as an update to the state. Notice that the `Node` does not need to return the whole `State` schema - just an update. After applying this update, the `State` would then be `{"foo": 2, "bar": ["hi"]}`. If the second node returns `{"bar": ["bye"]}` then the `State` would then be `{"foo": 2, "bar": ["hi", "bye"]}`. Notice here that the `bar` key is updated by adding the two lists together.

#### Working with Messages in Graph State

##### Why use messages?

Most modern LLM providers have a chat model interface that accepts a list of messages as input. LangChain's [`ChatModel`](https://python.langchain.com/docs/concepts/#chat-models) in particular accepts a list of `Message` objects as inputs. These messages come in a variety of forms such as `HumanMessage` (user input) or `AIMessage` (LLM response). To read more about what message objects are, please refer to [this](https://python.langchain.com/docs/concepts/#messages) conceptual guide.

##### Using Messages in your Graph

In many cases, it is helpful to store prior conversation history as a list of messages in your graph state. To do so, we can add a key (channel) to the graph state that stores a list of `Message` objects and annotate it with a reducer function (see `messages` key in the example below). The reducer function is vital to telling the graph how to update the list of `Message` objects in the state with each state update (for example, when a node sends an update). If you don't specify a reducer, every state update will overwrite the list of messages with the most recently provided value. If you wanted to simply append messages to the existing list, you could use `operator.add` as a reducer.

However, you might also want to manually update messages in your graph state (e.g. human-in-the-loop). If you were to use `operator.add`, the manual state updates you send to the graph would be appended to the existing list of messages, instead of updating existing messages. To avoid that, you need a reducer that can keep track of message IDs and overwrite existing messages, if updated. To achieve this, you can use the prebuilt `add_messages` function. For brand new messages, it will simply append to existing list, but it will also handle the updates for existing messages correctly.

##### Serialization

In addition to keeping track of message IDs, the `add_messages` function will also try to deserialize messages into LangChain `Message` objects whenever a state update is received on the `messages` channel. See more information on LangChain serialization/deserialization [here](https://python.langchain.com/docs/how_to/serialization/). This allows sending graph inputs / state updates in the following format:

```python
## this is supported
{"messages": [HumanMessage(content="message")]}

## and this is also supported
{"messages": [{"type": "human", "content": "message"}]}
```

Since the state updates are always deserialized into LangChain `Messages` when using `add_messages`, you should use dot notation to access message attributes, like `state["messages"][-1].content`. Below is an example of a graph that uses `add_messages` as it's reducer function.

```python
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

##### MessagesState

Since having a list of messages in your state is so common, there exists a prebuilt state called `MessagesState` which makes it easy to use messages. `MessagesState` is defined with a single `messages` key which is a list of `AnyMessage` objects and uses the `add_messages` reducer. Typically, there is more state to track than just messages, so we see people subclass this state and add more fields, like:

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    documents: list[str]
```

### Nodes

In LangGraph, nodes are typically python functions (sync or async) where the **first** positional argument is the [state](#state), and (optionally), the **second** positional argument is a "config", containing optional [configurable parameters](#configuration) (such as a `thread_id`).

Similar to `NetworkX`, you add these nodes to a graph using the [add_node][langgraph.graph.StateGraph.add_node] method:

```python
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

builder = StateGraph(dict)


def my_node(state: dict, config: RunnableConfig):
    print("In node: ", config["configurable"]["user_id"])
    return {"results": f"Hello, {state['input']}!"}


## The second argument is optional
def my_other_node(state: dict):
    return state


builder.add_node("my_node", my_node)
builder.add_node("other_node", my_other_node)
...
```

Behind the scenes, functions are converted to [RunnableLambda's](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableLambda.html#langchain_core.runnables.base.RunnableLambda), which add batch and async support to your function, along with native tracing and debugging.

If you add a node to graph without specifying a name, it will be given a default name equivalent to the function name.

```python
builder.add_node(my_node)
## You can then create edges to/from this node by referencing it as `"my_node"`
```

#### `START` Node

The `START` Node is a special node that represents the node sends user input to the graph. The main purpose for referencing this node is to determine which nodes should be called first.

```python
from langgraph.graph import START

graph.add_edge(START, "node_a")
```

#### `END` Node

The `END` Node is a special node that represents a terminal node. This node is referenced when you want to denote which edges have no actions after they are done.

```
from langgraph.graph import END

graph.add_edge("node_a", END)
```

### Edges

Edges define how the logic is routed and how the graph decides to stop. This is a big part of how your agents work and how different nodes communicate with each other. There are a few key types of edges:

- Normal Edges: Go directly from one node to the next.
- Conditional Edges: Call a function to determine which node(s) to go to next.
- Entry Point: Which node to call first when user input arrives.
- Conditional Entry Point: Call a function to determine which node(s) to call first when user input arrives.

A node can have MULTIPLE outgoing edges. If a node has multiple out-going edges, **all** of those destination nodes will be executed in parallel as a part of the next superstep.

#### Normal Edges

If you **always** want to go from node A to node B, you can use the [add_edge][langgraph.graph.StateGraph.add_edge] method directly.

```python
graph.add_edge("node_a", "node_b")
```

#### Conditional Edges

If you want to **optionally** route to 1 or more edges (or optionally terminate), you can use the [add_conditional_edges][langgraph.graph.StateGraph.add_conditional_edges] method. This method accepts the name of a node and a "routing function" to call after that node is executed:

```python
graph.add_conditional_edges("node_a", routing_function)
```

Similar to nodes, the `routing_function` accept the current `state` of the graph and return a value.

By default, the return value `routing_function` is used as the name of the node (or a list of nodes) to send the state to next. All those nodes will be run in parallel as a part of the next superstep.

You can optionally provide a dictionary that maps the `routing_function`'s output to the name of the next node.

```python
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
```

!!! tip
    Use [`Command`](#command) instead of conditional edges if you want to combine state updates and routing in a single function.

#### Entry Point

The entry point is the first node(s) that are run when the graph starts. You can use the [`add_edge`][langgraph.graph.StateGraph.add_edge] method from the virtual [`START`][langgraph.constants.START] node to the first node to execute to specify where to enter the graph.

```python
from langgraph.graph import START

graph.add_edge(START, "node_a")
```

#### Conditional Entry Point

A conditional entry point lets you start at different nodes depending on custom logic. You can use [`add_conditional_edges`][langgraph.graph.StateGraph.add_conditional_edges] from the virtual [`START`][langgraph.constants.START] node to accomplish this.

```python
from langgraph.graph import START

graph.add_conditional_edges(START, routing_function)
```

You can optionally provide a dictionary that maps the `routing_function`'s output to the name of the next node.

```python
graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})
```

### `Send`

By default, `Nodes` and `Edges` are defined ahead of time and operate on the same shared state. However, there can be cases where the exact edges are not known ahead of time and/or you may want different versions of `State` to exist at the same time. A common of example of this is with `map-reduce` design patterns. In this design pattern, a first node may generate a list of objects, and you may want to apply some other node to all those objects. The number of objects may be unknown ahead of time (meaning the number of edges may not be known) and the input `State` to the downstream `Node` should be different (one for each generated object).

To support this design pattern, LangGraph supports returning [`Send`][langgraph.types.Send] objects from conditional edges. `Send` takes two arguments: first is the name of the node, and second is the state to pass to that node.

```python
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

graph.add_conditional_edges("node_a", continue_to_jokes)
```

### `Command`

It can be useful to combine control flow (edges) and state updates (nodes). For example, you might want to BOTH perform state updates AND decide which node to go to next in the SAME node. LangGraph provides a way to do so by returning a [`Command`][langgraph.types.Command] object from node functions:

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

With `Command` you can also achieve dynamic control flow behavior (identical to [conditional edges](#conditional-edges)):

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")
```

!!! important

    When returning `Command` in your node functions, you must add return type annotations with the list of node names the node is routing to, e.g. `Command[Literal["my_other_node"]]`. This is necessary for the graph rendering and tells LangGraph that `my_node` can navigate to `my_other_node`.

Check out this [how-to guide](../how-tos/command.ipynb) for an end-to-end example of how to use `Command`.

#### When should I use Command instead of conditional edges?

Use `Command` when you need to **both** update the graph state **and** route to a different node. For example, when implementing [multi-agent handoffs](#handoffs) where it's important to route to a different agent and pass some information to that agent.

Use [conditional edges](#conditional-edges) to route between nodes conditionally without updating the state.

#### Using inside tools

A common use case is updating graph state from inside a tool. For example, in a customer support application you might want to look up customer information based on their account number or ID in the beginning of the conversation. To update the graph state from the tool, you can return `Command(update={"my_custom_key": "foo", "messages": [...]})` from the tool:

```python
@tool
def lookup_user_info(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig):
    """Use this to look up user information to better assist them with their questions."""
    user_info = get_user_info(config.get("configurable", {}).get("user_id"))
    return Command(
        update={
            # update the state keys
            "user_info": user_info,
            # update the message history
            "messages": [ToolMessage("Successfully looked up user information", tool_call_id=tool_call_id)]
        }
    )
```

!!! important
    You MUST include `messages` (or any state key used for the message history) in `Command.update` when returning `Command` from a tool and the list of messages in `messages` MUST contain a `ToolMessage`. This is necessary for the resulting message history to be valid (LLM providers require AI messages with tool calls to be followed by the tool result messages).

If you are using tools that update state via `Command`, we recommend using prebuilt [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] which automatically handles tools returning `Command` objects and propagates them to the graph state. If you're writing a custom node that calls tools, you would need to manually propagate `Command` objects returned by the tools as the update from node.

#### Human-in-the-loop

`Command` is an important part of human-in-the-loop workflows: when using `interrupt()` to collect user input, `Command` is then used to supply the input and resume execution via `Command(resume="User input")`. Check out [this conceptual guide](#docsconceptshuman_in_the_loopmd) for more information.

### Persistence

LangGraph provides built-in persistence for your agent's state using [checkpointers][langgraph.checkpoint.base.BaseCheckpointSaver]. Checkpointers save snapshots of the graph state at every superstep, allowing resumption at any time. This enables features like human-in-the-loop interactions, memory management, and fault-tolerance. You can even directly manipulate a graph's state after its execution using the
appropriate `get` and `update` methods. For more details, see the [persistence conceptual guide](#docsconceptspersistencemd).

### Threads

Threads in LangGraph represent individual sessions or conversations between your graph and a user. When using checkpointing, turns in a single conversation (and even steps within a single graph execution) are organized by a unique thread ID.

### Storage

LangGraph provides built-in document storage through the [BaseStore][langgraph.store.base.BaseStore] interface. Unlike checkpointers, which save state by thread ID, stores use custom namespaces for organizing data. This enables cross-thread persistence, allowing agents to maintain long-term memories, learn from past interactions, and accumulate knowledge over time. Common use cases include storing user profiles, building knowledge bases, and managing global preferences across all threads.

### Graph Migrations

LangGraph can easily handle migrations of graph definitions (nodes, edges, and state) even when using a checkpointer to track state.

- For threads at the end of the graph (i.e. not interrupted) you can change the entire topology of the graph (i.e. all nodes and edges, remove, add, rename, etc)
- For threads currently interrupted, we support all topology changes other than renaming / removing nodes (as that thread could now be about to enter a node that no longer exists) -- if this is a blocker please reach out and we can prioritize a solution.
- For modifying state, we have full backwards and forwards compatibility for adding and removing keys
- State keys that are renamed lose their saved state in existing threads
- State keys whose types change in incompatible ways could currently cause issues in threads with state from before the change -- if this is a blocker please reach out and we can prioritize a solution.

### Configuration

When creating a graph, you can also mark that certain parts of the graph are configurable. This is commonly done to enable easily switching between models or system prompts. This allows you to create a single "cognitive architecture" (the graph) but have multiple different instance of it.

You can optionally specify a `config_schema` when creating a graph.

```python
class ConfigSchema(TypedDict):
    llm: str

graph = StateGraph(State, config_schema=ConfigSchema)
```

You can then pass this configuration into the graph using the `configurable` config field.

```python
config = {"configurable": {"llm": "anthropic"}}

graph.invoke(inputs, config=config)
```

You can then access and use this configuration inside a node:

```python
def node_a(state, config):
    llm_type = config.get("configurable", {}).get("llm", "openai")
    llm = get_llm(llm_type)
    ...
```

See [this guide](../how-tos/configuration.ipynb) for a full breakdown on configuration.

#### Recursion Limit

The recursion limit sets the maximum number of [super-steps](#graphs) the graph can execute during a single execution. Once the limit is reached, LangGraph will raise `GraphRecursionError`. By default this value is set to 25 steps. The recursion limit can be set on any graph at runtime, and is passed to `.invoke`/`.stream` via the config dictionary. Importantly, `recursion_limit` is a standalone `config` key and should not be passed inside the `configurable` key as all other user-defined configuration. See the example below:

```python
graph.invoke(inputs, config={"recursion_limit": 5, "configurable":{"llm": "anthropic"}})
```

Read [this how-to](https://langchain-ai.github.io/langgraph/how-tos/recursion-limit/) to learn more about how the recursion limit works.

### `interrupt`

Use the [interrupt](#langgraph.types.interrupt) function to **pause** the graph at specific points to collect user input. The `interrupt` function surfaces interrupt information to the client, allowing the developer to collect user input, validate the graph state, or make decisions before resuming execution.

```python
from langgraph.types import interrupt

def human_approval_node(state: State):
    ...
    answer = interrupt(
        # This value will be sent to the client.
        # It can be any JSON serializable value.
        {"question": "is it ok to continue?"},
    )
    ...
```

Resuming the graph is done by passing a [`Command`](#command) object to the graph with the `resume` key set to the value returned by the `interrupt` function.

Read more about how the `interrupt` is used for **human-in-the-loop** workflows in the [Human-in-the-loop conceptual guide](#docsconceptshuman_in_the_loopmd).

### Breakpoints

Breakpoints pause graph execution at specific points and enable stepping through execution step by step. Breakpoints are powered by LangGraph's [**persistence layer**](#docsconceptspersistencemd), which saves the state after each graph step. Breakpoints can also be used to enable [**human-in-the-loop**](#docsconceptshuman_in_the_loopmd) workflows, though we recommend using the [`interrupt` function](#interrupt-function) for this purpose.

Read more about breakpoints in the [Breakpoints conceptual guide](#docsconceptsbreakpointsmd).

### Subgraphs

A subgraph is a [graph](#graphs) that is used as a [node](#nodes) in another graph. This is nothing more than the age-old concept of encapsulation, applied to LangGraph. Some reasons for using subgraphs are:

- building [multi-agent systems](#docsconceptsmulti_agentmd)

- when you want to reuse a set of nodes in multiple graphs, which maybe share some state, you can define them once in a subgraph and then use them in multiple parent graphs

- when you want different teams to work on different parts of the graph independently, you can define each part as a subgraph, and as long as the subgraph interface (the input and output schemas) is respected, the parent graph can be built without knowing any details of the subgraph

There are two ways to add subgraphs to a parent graph:

- add a node with the compiled subgraph: this is useful when the parent graph and the subgraph share state keys and you don't need to transform state on the way in or out

```python
builder.add_node("subgraph", subgraph_builder.compile())
```

- add a node with a function that invokes the subgraph: this is useful when the parent graph and the subgraph have different state schemas and you need to transform state before or after calling the subgraph

```python
subgraph = subgraph_builder.compile()

def call_subgraph(state: State):
    return subgraph.invoke({"subgraph_key": state["parent_key"]})

builder.add_node("subgraph", call_subgraph)
```

Let's take a look at examples for each.

#### As a compiled graph

The simplest way to create subgraph nodes is by using a [compiled subgraph](#compiling-your-graph) directly. When doing so, it is **important** that the parent graph and the subgraph [state schemas](#state) share at least one key which they can use to communicate. If your graph and subgraph do not share any keys, you should use write a function [invoking the subgraph](#as-a-function) instead.

!!! Note
    If you pass extra keys to the subgraph node (i.e., in addition to the shared keys), they will be ignored by the subgraph node. Similarly, if you return extra keys from the subgraph, they will be ignored by the parent graph.

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str

## Define subgraph
def subgraph_node(state: SubgraphState):
    # note that this subgraph node can communicate with the parent graph via the shared "foo" key
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node)
...
subgraph = subgraph_builder.compile()

## Define parent graph
builder = StateGraph(State)
builder.add_node("subgraph", subgraph)
...
graph = builder.compile()
```

#### As a function

You might want to define a subgraph with a completely different schema. In this case, you can create a node function that invokes the subgraph. This function will need to [transform](../how-tos/subgraph-transform-state.ipynb) the input (parent) state to the subgraph state before invoking the subgraph, and transform the results back to the parent state before returning the state update from the node.

```python
class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    # note that none of these keys are shared with the parent graph state
    bar: str
    baz: str

## Define subgraph
def subgraph_node(state: SubgraphState):
    return {"bar": state["bar"] + "baz"}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node)
...
subgraph = subgraph_builder.compile()

## Define parent graph
def node(state: State):
    # transform the state to the subgraph state
    response = subgraph.invoke({"bar": state["foo"]})
    # transform response back to the parent state
    return {"foo": response["bar"]}

builder = StateGraph(State)
## note that we are using `node` function instead of a compiled subgraph
builder.add_node(node)
...
graph = builder.compile()
```

### Visualization

It's often nice to be able to visualize graphs, especially as they get more complex. LangGraph comes with several built-in ways to visualize graphs. See [this how-to guide](../how-tos/visualization.ipynb) for more info.

### Streaming

LangGraph is built with first class support for streaming, including streaming updates from graph nodes during the execution, streaming tokens from LLM calls and more. See this [conceptual guide](#docsconceptsstreamingmd) for more information.


<a name="docsconceptsmemorymd"></a>

## Memory

### What is Memory?

[Memory](https://pmc.ncbi.nlm.nih.gov/articles/PMC10410470/) is a cognitive function that allows people to store, retrieve, and use information to understand their present and future. Consider the frustration of working with a colleague who forgets everything you tell them, requiring constant repetition! As AI agents undertake more complex tasks involving numerous user interactions, equipping them with memory becomes equally crucial for efficiency and user satisfaction. With memory, agents can learn from feedback and adapt to users' preferences. This guide covers two types of memory based on recall scope:

**Short-term memory**, or [thread](#threads)-scoped memory, can be recalled at any time **from within** a single conversational thread with a user. LangGraph manages short-term memory as a part of your agent's [state](#state). State is persisted to a database using a [checkpointer](#checkpoints) so the thread can be resumed at any time. Short-term memory updates when the graph is invoked or a step is completed, and the State is read at the start of each step.

**Long-term memory** is shared **across** conversational threads. It can be recalled _at any time_ and **in any thread**. Memories are scoped to any custom namespace, not just within a single thread ID. LangGraph provides [stores](#memory-store) ([reference doc](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore)) to let you save and recall long-term memories.

Both are important to understand and implement for your application.

![](img/memory/short-vs-long.png)

### Short-term memory

Short-term memory lets your application remember previous interactions within a single [thread](#threads) or conversation. A [thread](#threads) organizes multiple interactions in a session, similar to the way email groups messages in a single conversation.

LangGraph manages short-term memory as part of the agent's state, persisted via thread-scoped checkpoints. This state can normally include the conversation history along with other stateful data, such as uploaded files, retrieved documents, or generated artifacts. By storing these in the graph's state, the bot can access the full context for a given conversation while maintaining separation between different threads.

Since conversation history is the most common form of representing short-term memory, in the next section, we will cover techniques for managing conversation history when the list of messages becomes **long**. If you want to stick to the high-level concepts, continue on to the [long-term memory](#long-term-memory) section.

#### Managing long conversation history

Long conversations pose a challenge to today's LLMs. The full history may not even fit inside an LLM's context window, resulting in an irrecoverable error. Even _if_ your LLM technically supports the full context length, most LLMs still perform poorly over long contexts. They get "distracted" by stale or off-topic content, all while suffering from slower response times and higher costs.

Managing short-term memory is an exercise of balancing [precision & recall](https://en.wikipedia.org/wiki/Precision_and_recall#:~:text=Precision%20can%20be%20seen%20as,irrelevant%20ones%20are%20also%20returned) with your application's other performance requirements (latency & cost). As always, it's important to think critically about how you represent information for your LLM and to look at your data. We cover a few common techniques for managing message lists below and hope to provide sufficient context for you to pick the best tradeoffs for your application:

- [Editing message lists](#editing-message-lists): How to think about trimming and filtering a list of messages before passing to language model.
- [Summarizing past conversations](#summarizing-past-conversations): A common technique to use when you don't just want to filter the list of messages.

#### Editing message lists

Chat models accept context using [messages](https://python.langchain.com/docs/concepts/#messages), which include developer provided instructions (a system message) and user inputs (human messages). In chat applications, messages alternate between human inputs and model responses, resulting in a list of messages that grows longer over time. Because context windows are limited and token-rich message lists can be costly, many applications can benefit from using techniques to manually remove or forget stale information.

![](img/memory/filter.png)

The most direct approach is to remove old messages from a list (similar to a [least-recently used cache](https://en.wikipedia.org/wiki/Page_replacement_algorithm#Least_recently_used)).

The typical technique for deleting content from a list in LangGraph is to return an update from a node telling the system to delete some portion of the list. You get to define what this update looks like, but a common approach would be to let you return an object or dictionary specifying which values to retain.

```python
def manage_list(existing: list, updates: Union[list, dict]):
    if isinstance(updates, list):
        # Normal case, add to the history
        return existing + updates
    elif isinstance(updates, dict) and updates["type"] == "keep":
        # You get to decide what this looks like.
        # For example, you could simplify and just accept a string "DELETE"
        # and clear the entire list.
        return existing[updates["from"]:updates["to"]]
    # etc. We define how to interpret updates

class State(TypedDict):
    my_list: Annotated[list, manage_list]

def my_node(state: State):
    return {
        # We return an update for the field "my_list" saying to
        # keep only values from index -5 to the end (deleting the rest)
        "my_list": {"type": "keep", "from": -5, "to": None}
    }
```

LangGraph will call the `manage_list` "[reducer](#reducers)" function any time an update is returned under the key "my_list". Within that function, we define what types of updates to accept. Typically, messages will be added to the existing list (the conversation will grow); however, we've also added support to accept a dictionary that lets you "keep" certain parts of the state. This lets you programmatically drop old message context.

Another common approach is to let you return a list of "remove" objects that specify the IDs of all messages to delete. If you're using the LangChain messages and the [`add_messages`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages) reducer (or `MessagesState`, which uses the same underlying functionality) in LangGraph, you can do this using a `RemoveMessage`.

```python
from langchain_core.messages import RemoveMessage, AIMessage
from langgraph.graph import add_messages
## ... other imports

class State(TypedDict):
    # add_messages will default to upserting messages by ID to the existing list
    # if a RemoveMessage is returned, it will delete the message in the list by ID
    messages: Annotated[list, add_messages]

def my_node_1(state: State):
    # Add an AI message to the `messages` list in the state
    return {"messages": [AIMessage(content="Hi")]}

def my_node_2(state: State):
    # Delete all but the last 2 messages from the `messages` list in the state
    delete_messages = [RemoveMessage(id=m.id) for m in state['messages'][:-2]]
    return {"messages": delete_messages}

```

In the example above, the `add_messages` reducer allows us to [append](https://langchain-ai.github.io/langgraph/concepts/low_level/#serialization) new messages to the `messages` state key as shown in `my_node_1`. When it sees a `RemoveMessage`, it will delete the message with that ID from the list (and the RemoveMessage will then be discarded). For more information on LangChain-specific message handling, check out [this how-to on using `RemoveMessage` ](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/).

See this how-to [guide](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/) and module 2 from our [LangChain Academy](https://github.com/langchain-ai/langchain-academy/tree/main/module-2) course for example usage.

#### Summarizing past conversations

The problem with trimming or removing messages, as shown above, is that we may lose information from culling of the message queue. Because of this, some applications benefit from a more sophisticated approach of summarizing the message history using a chat model.

![](img/memory/summary.png)

Simple prompting and orchestration logic can be used to achieve this. As an example, in LangGraph we can extend the [MessagesState](https://langchain-ai.github.io/langgraph/concepts/low_level/#working-with-messages-in-graph-state) to include a `summary` key.

```python
from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str
```

Then, we can generate a summary of the chat history, using any existing summary as context for the next summary. This `summarize_conversation` node can be called after some number of messages have accumulated in the `messages` state key.

```python
def summarize_conversation(state: State):

    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
```

See this how-to [here](https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/) and module 2 from our [LangChain Academy](https://github.com/langchain-ai/langchain-academy/tree/main/module-2) course for example usage.

#### Knowing **when** to remove messages

Most LLMs have a maximum supported context window (denominated in tokens). A simple way to decide when to truncate messages is to count the tokens in the message history and truncate whenever it approaches that limit. Naive truncation is straightforward to implement on your own, though there are a few "gotchas". Some model APIs further restrict the sequence of message types (must start with human message, cannot have consecutive messages of the same type, etc.). If you're using LangChain, you can use the [`trim_messages`](https://python.langchain.com/docs/how_to/trim_messages/#trimming-based-on-token-count) utility and specify the number of tokens to keep from the list, as well as the `strategy` (e.g., keep the last `max_tokens`) to use for handling the boundary.

Below is an example.

```python
from langchain_core.messages import trim_messages
trim_messages(
    messages,
    # Keep the last <= n_count tokens of the messages.
    strategy="last",
    # Remember to adjust based on your model
    # or else pass a custom token_encoder
    token_counter=ChatOpenAI(model="gpt-4"),
    # Remember to adjust based on the desired conversation
    # length
    max_tokens=45,
    # Most chat models expect that chat history starts with either:
    # (1) a HumanMessage or
    # (2) a SystemMessage followed by a HumanMessage
    start_on="human",
    # Most chat models expect that chat history ends with either:
    # (1) a HumanMessage or
    # (2) a ToolMessage
    end_on=("human", "tool"),
    # Usually, we want to keep the SystemMessage
    # if it's present in the original history.
    # The SystemMessage has special instructions for the model.
    include_system=True,
)
```

### Long-term memory

Long-term memory in LangGraph allows systems to retain information across different conversations or sessions. Unlike short-term memory, which is **thread-scoped**, long-term memory is saved within custom "namespaces."

#### Storing memories

LangGraph stores long-term memories as JSON documents in a [store](#memory-store) ([reference doc](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore)). Each memory is organized under a custom `namespace` (similar to a folder) and a distinct `key` (like a filename). Namespaces often include user or org IDs or other labels that makes it easier to organize information. This structure enables hierarchical organization of memories. Cross-namespace searching is then supported through content filters. See the example below for an example.

```python
from langgraph.store.memory import InMemoryStore


def embed(texts: list[str]) -> list[list[float]]:
    # Replace with an actual embedding function or LangChain embeddings object
    return [[1.0, 2.0] * len(texts)]


## InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
store = InMemoryStore(index={"embed": embed, "dims": 2})
user_id = "my-user"
application_context = "chitchat"
namespace = (user_id, application_context)
store.put(
    namespace,
    "a-memory",
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "my-key": "my-value",
    },
)
## get the "memory" by ID
item = store.get(namespace, "a-memory")
## search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
items = store.search(
    namespace, filter={"my-key": "my-value"}, query="language preferences"
)
```

#### Framework for thinking about long-term memory

Long-term memory is a complex challenge without a one-size-fits-all solution. However, the following questions provide a structure framework to help you navigate the different techniques:

**What is the type of memory?**

Humans use memories to remember [facts](https://en.wikipedia.org/wiki/Semantic_memory), [experiences](https://en.wikipedia.org/wiki/Episodic_memory), and [rules](https://en.wikipedia.org/wiki/Procedural_memory). AI agents can use memory in the same ways. For example, AI agents can use memory to remember specific facts about a user to accomplish a task. We expand on several types of memories in the [section below](#memory-types).

**When do you want to update memories?**

Memory can be updated as part of an agent's application logic (e.g. "on the hot path"). In this case, the agent typically decides to remember facts before responding to a user. Alternatively, memory can be updated as a background task (logic that runs in the background / asynchronously and generates memories). We explain the tradeoffs between these approaches in the [section below](#writing-memories).

### Memory types

Different applications require various types of memory. Although the analogy isn't perfect, examining [human memory types](https://www.psychologytoday.com/us/basics/memory/types-of-memory?ref=blog.langchain.dev) can be insightful. Some research (e.g., the [CoALA paper](https://arxiv.org/pdf/2309.02427)) have even mapped these human memory types to those used in AI agents.

| Memory Type | What is Stored | Human Example | Agent Example |
|-------------|----------------|---------------|---------------|
| Semantic | Facts | Things I learned in school | Facts about a user |
| Episodic | Experiences | Things I did | Past agent actions |
| Procedural | Instructions | Instincts or motor skills | Agent system prompt |

#### Semantic Memory

[Semantic memory](https://en.wikipedia.org/wiki/Semantic_memory), both in humans and AI agents, involves the retention of specific facts and concepts. In humans, it can include information learned in school and the understanding of concepts and their relationships. For AI agents, semantic memory is often used to personalize applications by remembering facts or concepts from past interactions. 

> Note: Not to be confused with "semantic search" which is a technique for finding similar content using "meaning" (usually as embeddings). Semantic memory is a term from psychology, referring to storing facts and knowledge, while semantic search is a method for retrieving information based on meaning rather than exact matches.


##### Profile

Semantic memories can be managed in different ways. For example, memories can be a single, continuously updated "profile" of well-scoped and specific information about a user, organization, or other entity (including the agent itself). A profile is generally just a JSON document with various key-value pairs you've selected to represent your domain. 

When remembering a profile, you will want to make sure that you are **updating** the profile each time. As a result, you will want to pass in the previous profile and [ask the model to generate a new profile](https://github.com/langchain-ai/memory-template) (or some [JSON patch](https://github.com/hinthornw/trustcall) to apply to the old profile). This can be become error-prone as the profile gets larger, and may benefit from splitting a profile into multiple documents or **strict** decoding when generating documents to ensure the memory schemas remains valid.

![](img/memory/update-profile.png)

##### Collection

Alternatively, memories can be a collection of documents that are continuously updated and extended over time. Each individual memory can be more narrowly scoped and easier to generate, which means that you're less likely to **lose** information over time. It's easier for an LLM to generate _new_ objects for new information than reconcile new information with an existing profile. As a result, a document collection tends to lead to [higher recall downstream](https://en.wikipedia.org/wiki/Precision_and_recall).

However, this shifts some complexity memory updating. The model must now _delete_ or _update_ existing items in the list, which can be tricky. In addition, some models may default to over-inserting and others may default to over-updating. See the [Trustcall](https://github.com/hinthornw/trustcall) package for one way to manage this and consider evaluation (e.g., with a tool like [LangSmith](https://docs.smith.langchain.com/tutorials/Developers/evaluation)) to help you tune the behavior.

Working with document collections also shifts complexity to memory **search** over the list. The `Store` currently supports both [semantic search](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.SearchOp.query) and [filtering by content](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.SearchOp.filter).

Finally, using a collection of memories can make it challenging to provide comprehensive context to the model. While individual memories may follow a specific schema, this structure might not capture the full context or relationships between memories. As a result, when using these memories to generate responses, the model may lack important contextual information that would be more readily available in a unified profile approach.

![](img/memory/update-list.png)

Regardless of memory management approach, the central point is that the agent will use the semantic memories to [ground its responses](https://python.langchain.com/docs/concepts/rag/), which often leads to more personalized and relevant interactions.

#### Episodic Memory

[Episodic memory](https://en.wikipedia.org/wiki/Episodic_memory), in both humans and AI agents, involves recalling past events or actions. The [CoALA paper](https://arxiv.org/pdf/2309.02427) frames this well: facts can be written to semantic memory, whereas *experiences* can be written to episodic memory. For AI agents, episodic memory is often used to help an agent remember how to accomplish a task. 

In practice, episodic memories are often implemented through [few-shot example prompting](https://python.langchain.com/docs/concepts/few_shot_prompting/), where agents learn from past sequences to perform tasks correctly. Sometimes it's easier to "show" than "tell" and LLMs learn well from examples. Few-shot learning lets you ["program"](https://x.com/karpathy/status/1627366413840322562) your LLM by updating the prompt with input-output examples to illustrate the intended behavior. While various [best-practices](https://python.langchain.com/docs/concepts/#1-generating-examples) can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input.

Note that the memory [store](#memory-store) is just one way to store data as few-shot examples. If you want to have more developer involvement, or tie few-shots more closely to your evaluation harness, you can also use a [LangSmith Dataset](https://docs.smith.langchain.com/evaluation/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) to store your data. Then dynamic few-shot example selectors can be used out-of-the box to achieve this same goal. LangSmith will index the dataset for you and enable retrieval of few shot examples that are most relevant to the user input based upon keyword similarity ([using a BM25-like algorithm](https://docs.smith.langchain.com/how_to_guides/datasets/index_datasets_for_dynamic_few_shot_example_selection) for keyword based similarity). 

See this how-to [video](https://www.youtube.com/watch?v=37VaU7e7t5o) for example usage of dynamic few-shot example selection in LangSmith. Also, see this [blog post](https://blog.langchain.dev/few-shot-prompting-to-improve-tool-calling-performance/) showcasing few-shot prompting to improve tool calling performance and this [blog post](https://blog.langchain.dev/aligning-llm-as-a-judge-with-human-preferences/) using few-shot example to align an LLMs to human preferences.

#### Procedural Memory

[Procedural memory](https://en.wikipedia.org/wiki/Procedural_memory), in both humans and AI agents, involves remembering the rules used to perform tasks. In humans, procedural memory is like the internalized knowledge of how to perform tasks, such as riding a bike via basic motor skills and balance. Episodic memory, on the other hand, involves recalling specific experiences, such as the first time you successfully rode a bike without training wheels or a memorable bike ride through a scenic route. For AI agents, procedural memory is a combination of model weights, agent code, and agent's prompt that collectively determine the agent's functionality. 

In practice, it is fairly uncommon for agents to modify their model weights or rewrite their code. However, it is more common for agents to [modify their own prompts](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-generator). 

One effective approach to refining an agent's instructions is through ["Reflection"](https://blog.langchain.dev/reflection-agents/) or meta-prompting. This involves prompting the agent with its current instructions (e.g., the system prompt) along with recent conversations or explicit user feedback. The agent then refines its own instructions based on this input. This method is particularly useful for tasks where instructions are challenging to specify upfront, as it allows the agent to learn and adapt from its interactions.

For example, we built a [Tweet generator](https://www.youtube.com/watch?v=Vn8A3BxfplE) using external feedback and prompt re-writing to produce high-quality paper summaries for Twitter. In this case, the specific summarization prompt was difficult to specify *a priori*, but it was fairly easy for a user to critique the generated Tweets and provide feedback on how to improve the summarization process. 

The below pseudo-code shows how you might implement this with the LangGraph memory [store](#memory-store), using the store to save a prompt, the `update_instructions` node to get the current prompt (as well as feedback from the conversation with the user captured in `state["messages"]`), update the prompt, and save the new prompt back to the store. Then, the `call_model` get the updated prompt from the store and uses it to generate a response.

```python
## Node that *uses* the instructions
def call_model(state: State, store: BaseStore):
    namespace = ("agent_instructions", )
    instructions = store.get(namespace, key="agent_a")[0]
    # Application logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"])
    ...

## Node that updates instructions
def update_instructions(state: State, store: BaseStore):
    namespace = ("instructions",)
    current_instructions = store.search(namespace)[0]
    # Memory logic
    prompt = prompt_template.format(instructions=instructions.value["instructions"], conversation=state["messages"])
    output = llm.invoke(prompt)
    new_instructions = output['new_instructions']
    store.put(("agent_instructions",), "agent_a", {"instructions": new_instructions})
    ...
```

![](img/memory/update-instructions.png)

### Writing memories

While [humans often form long-term memories during sleep](https://medicine.yale.edu/news-article/sleeps-crucial-role-in-preserving-memory/), AI agents need a different approach. When and how should agents create new memories? There are at least two primary methods for agents to write memories: "on the hot path" and "in the background".

![](img/memory/hot_path_vs_background.png)

#### Writing memories in the hot path

Creating memories during runtime offers both advantages and challenges. On the positive side, this approach allows for real-time updates, making new memories immediately available for use in subsequent interactions. It also enables transparency, as users can be notified when memories are created and stored.

However, this method also presents challenges. It may increase complexity if the agent requires a new tool to decide what to commit to memory. In addition, the process of reasoning about what to save to memory can impact agent latency. Finally, the agent must multitask between memory creation and its other responsibilities, potentially affecting the quantity and quality of memories created.

As an example, ChatGPT uses a [save_memories](https://openai.com/index/memory-and-new-controls-for-chatgpt/) tool to upsert memories as content strings, deciding whether and how to use this tool with each user message. See our [memory-agent](https://github.com/langchain-ai/memory-agent) template as an reference implementation.

#### Writing memories in the background

Creating memories as a separate background task offers several advantages. It eliminates latency in the primary application, separates application logic from memory management, and allows for more focused task completion by the agent. This approach also provides flexibility in timing memory creation to avoid redundant work.

However, this method has its own challenges. Determining the frequency of memory writing becomes crucial, as infrequent updates may leave other threads without new context. Deciding when to trigger memory formation is also important. Common strategies include scheduling after a set time period (with rescheduling if new events occur), using a cron schedule, or allowing manual triggers by users or the application logic.

See our [memory-service](https://github.com/langchain-ai/memory-template) template as an reference implementation.


<a name="docsconceptsmulti_agentmd"></a>

## Multi-agent Systems

An [agent](#agent-architectures) is _a system that uses an LLM to decide the control flow of an application_. As you develop these systems, they might grow more complex over time, making them harder to manage and scale. For example, you might run into the following problems:

- agent has too many tools at its disposal and makes poor decisions about which tool to call next
- context grows too complex for a single agent to keep track of
- there is a need for multiple specialization areas in the system (e.g. planner, researcher, math expert, etc.)

To tackle these, you might consider breaking your application into multiple smaller, independent agents and composing them into a **multi-agent system**. These independent agents can be as simple as a prompt and an LLM call, or as complex as a [ReAct](#react-implementation) agent (and more!).

The primary benefits of using multi-agent systems are:

- **Modularity**: Separate agents make it easier to develop, test, and maintain agentic systems.
- **Specialization**: You can create expert agents focused on specific domains, which helps with the overall system performance.
- **Control**: You can explicitly control how agents communicate (as opposed to relying on function calling).

### Multi-agent architectures

![](./img/multi_agent/architectures.png)

There are several ways to connect agents in a multi-agent system:

- **Network**: each agent can communicate with [every other agent](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/). Any agent can decide which other agent to call next.
- **Supervisor**: each agent communicates with a single [supervisor](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/) agent. Supervisor agent makes decisions on which agent should be called next.
- **Supervisor (tool-calling)**: this is a special case of supervisor architecture. Individual agents can be represented as tools. In this case, a supervisor agent uses a tool-calling LLM to decide which of the agent tools to call, as well as the arguments to pass to those agents.
- **Hierarchical**: you can define a multi-agent system with [a supervisor of supervisors](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/). This is a generalization of the supervisor architecture and allows for more complex control flows.
- **Custom multi-agent workflow**: each agent communicates with only a subset of agents. Parts of the flow are deterministic, and only some agents can decide which other agents to call next.

#### Handoffs

In multi-agent architectures, agents can be represented as graph nodes. Each agent node executes its step(s) and decides whether to finish execution or route to another agent, including potentially routing to itself (e.g., running in a loop). A common pattern in multi-agent interactions is handoffs, where one agent hands off control to another. Handoffs allow you to specify:

- __destination__: target agent to navigate to (e.g., name of the node to go to)
- __payload__: [information to pass to that agent](#communication-between-agents) (e.g., state update)

To implement handoffs in LangGraph, agent nodes can return [`Command`](#command) object that allows you to combine both control flow and state updates:

```python
def agent(state) -> Command[Literal["agent", "another_agent"]]:
    # the condition for routing/halting can be anything, e.g. LLM tool call / structured output, etc.
    goto = get_next_agent(...)  # 'agent' / 'another_agent'
    return Command(
        # Specify which agent to call next
        goto=goto,
        # Update the graph state
        update={"my_state_key": "my_state_value"}
    )
```

In a more complex scenario where each agent node is itself a graph (i.e., a [subgraph](#subgraphs)), a node in one of the agent subgraphs might want to navigate to a different agent. For example, if you have two agents, `alice` and `bob` (subgraph nodes in a parent graph), and `alice` needs to navigate to `bob`, you can set `graph=Command.PARENT` in the `Command` object:

```python
def some_node_inside_alice(state)
    return Command(
        goto="bob",
        update={"my_state_key": "my_state_value"},
        # specify which graph to navigate to (defaults to the current graph)
        graph=Command.PARENT,
    )
```

!!! note
    If you need to support visualization for subgraphs communicating using `Command(graph=Command.PARENT)` you would need to wrap them in a node function with `Command` annotation, e.g. instead of this:

    ```python
    builder.add_node(alice)
    ```

    you would need to do this:

    ```python
    def call_alice(state) -> Command[Literal["bob"]]:
        return alice.invoke(state)

    builder.add_node("alice", call_alice)
    ```

##### Handoffs as tools

One of the most common agent types is a ReAct-style tool-calling agents. For those types of agents, a common pattern is wrapping a handoff in a tool call, e.g.:

```python
def transfer_to_bob(state):
    """Transfer to bob."""
    return Command(
        goto="bob",
        update={"my_state_key": "my_state_value"},
        graph=Command.PARENT,
    )
```

This is a special case of updating the graph state from tools where in addition the state update, the control flow is included as well.

!!! important

    If you want to use tools that return `Command`, you can either use prebuilt [`create_react_agent`][langgraph.prebuilt.chat_agent_executor.create_react_agent] / [`ToolNode`][langgraph.prebuilt.tool_node.ToolNode] components, or implement your own tool-executing node that collects `Command` objects returned by the tools and returns a list of them, e.g.:
    
    ```python
    def call_tools(state):
        ...
        commands = [tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls]
        return commands
    ```

Let's now take a closer look at the different multi-agent architectures.

#### Network

In this architecture, agents are defined as graph nodes. Each agent can communicate with every other agent (many-to-many connections) and can decide which agent to call next. This architecture is good for problems that do not have a clear hierarchy of agents or a specific sequence in which agents should be called.


```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

model = ChatOpenAI()

def agent_1(state: MessagesState) -> Command[Literal["agent_2", "agent_3", END]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    response = model.invoke(...)
    # route to one of the agents or exit based on the LLM's decision
    # if the LLM returns "__end__", the graph will finish execution
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def agent_2(state: MessagesState) -> Command[Literal["agent_1", "agent_3", END]]:
    response = model.invoke(...)
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

def agent_3(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    ...
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )

builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)
builder.add_node(agent_3)

builder.add_edge(START, "agent_1")
network = builder.compile()
```

#### Supervisor

In this architecture, we define agents as nodes and add a supervisor node (LLM) that decides which agent nodes should be called next. We use [`Command`](#command) to route execution to the appropriate agent node based on supervisor's decision. This architecture also lends itself well to running multiple agents in parallel or using [map-reduce](../how-tos/map-reduce.ipynb) pattern.

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI()

def supervisor(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    response = model.invoke(...)
    # route to one of the agents or exit based on the supervisor's decision
    # if the supervisor returns "__end__", the graph will finish execution
    return Command(goto=response["next_agent"])

def agent_1(state: MessagesState) -> Command[Literal["supervisor"]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = model.invoke(...)
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

def agent_2(state: MessagesState) -> Command[Literal["supervisor"]]:
    response = model.invoke(...)
    return Command(
        goto="supervisor",
        update={"messages": [response]},
    )

builder = StateGraph(MessagesState)
builder.add_node(supervisor)
builder.add_node(agent_1)
builder.add_node(agent_2)

builder.add_edge(START, "supervisor")

supervisor = builder.compile()
```

Check out this [tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/) for an example of supervisor multi-agent architecture.

#### Supervisor (tool-calling)

In this variant of the [supervisor](#supervisor) architecture, we define individual agents as **tools** and use a tool-calling LLM in the supervisor node. This can be implemented as a [ReAct](#react-implementation)-style agent with two nodes ÔÇö an LLM node (supervisor) and a tool-calling node that executes tools (agents in this case).

```python
from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent

model = ChatOpenAI()

## this is the agent function that will be called as tool
## notice that you can pass the state to the tool via InjectedState annotation
def agent_1(state: Annotated[dict, InjectedState]):
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # and add any additional logic (different models, custom prompts, structured output, etc.)
    response = model.invoke(...)
    # return the LLM response as a string (expected tool response format)
    # this will be automatically turned to ToolMessage
    # by the prebuilt create_react_agent (supervisor)
    return response.content

def agent_2(state: Annotated[dict, InjectedState]):
    response = model.invoke(...)
    return response.content

tools = [agent_1, agent_2]
## the simplest way to build a supervisor w/ tool-calling is to use prebuilt ReAct agent graph
## that consists of a tool-calling LLM node (i.e. supervisor) and a tool-executing node
supervisor = create_react_agent(model, tools)
```

#### Hierarchical

As you add more agents to your system, it might become too hard for the supervisor to manage all of them. The supervisor might start making poor decisions about which agent to call next, the context might become too complex for a single supervisor to keep track of. In other words, you end up with the same problems that motivated the multi-agent architecture in the first place.

To address this, you can design your system _hierarchically_. For example, you can create separate, specialized teams of agents managed by individual supervisors, and a top-level supervisor to manage the teams.

```python
from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI()

## define team 1 (same as the single supervisor example above)

def team_1_supervisor(state: MessagesState) -> Command[Literal["team_1_agent_1", "team_1_agent_2", END]]:
    response = model.invoke(...)
    return Command(goto=response["next_agent"])

def team_1_agent_1(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_1_supervisor", update={"messages": [response]})

def team_1_agent_2(state: MessagesState) -> Command[Literal["team_1_supervisor"]]:
    response = model.invoke(...)
    return Command(goto="team_1_supervisor", update={"messages": [response]})

team_1_builder = StateGraph(Team1State)
team_1_builder.add_node(team_1_supervisor)
team_1_builder.add_node(team_1_agent_1)
team_1_builder.add_node(team_1_agent_2)
team_1_builder.add_edge(START, "team_1_supervisor")
team_1_graph = team_1_builder.compile()

## define team 2 (same as the single supervisor example above)
class Team2State(MessagesState):
    next: Literal["team_2_agent_1", "team_2_agent_2", "__end__"]

def team_2_supervisor(state: Team2State):
    ...

def team_2_agent_1(state: Team2State):
    ...

def team_2_agent_2(state: Team2State):
    ...

team_2_builder = StateGraph(Team2State)
...
team_2_graph = team_2_builder.compile()


## define top-level supervisor

builder = StateGraph(MessagesState)
def top_level_supervisor(state: MessagesState):
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which team to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_team" field)
    response = model.invoke(...)
    # route to one of the teams or exit based on the supervisor's decision
    # if the supervisor returns "__end__", the graph will finish execution
    return Command(goto=response["next_team"])

builder = StateGraph(MessagesState)
builder.add_node(top_level_supervisor)
builder.add_node(team_1_graph)
builder.add_node(team_2_graph)

builder.add_edge(START, "top_level_supervisor")
graph = builder.compile()
```

#### Custom multi-agent workflow

In this architecture we add individual agents as graph nodes and define the order in which agents are called ahead of time, in a custom workflow. In LangGraph the workflow can be defined in two ways:

- **Explicit control flow (normal edges)**: LangGraph allows you to explicitly define the control flow of your application (i.e. the sequence of how agents communicate) explicitly, via [normal graph edges](#normal-edges). This is the most deterministic variant of this architecture above ÔÇö we always know which agent will be called next ahead of time.

- **Dynamic control flow (Command)**: in LangGraph you can allow LLMs to decide parts of your application control flow. This can be achieved by using [`Command`](#command). A special case of this is a [supervisor tool-calling](#supervisor-tool-calling) architecture. In that case, the tool-calling LLM powering the supervisor agent will make decisions about the order in which the tools (agents) are being called.

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START

model = ChatOpenAI()

def agent_1(state: MessagesState):
    response = model.invoke(...)
    return {"messages": [response]}

def agent_2(state: MessagesState):
    response = model.invoke(...)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)
## define the flow explicitly
builder.add_edge(START, "agent_1")
builder.add_edge("agent_1", "agent_2")
```

### Communication between agents

The most important thing when building multi-agent systems is figuring out how the agents communicate. There are few different considerations:

- Do agents communicate via [**via graph state or via tool calls**](#graph-state-vs-tool-calls)?
- What if two agents have [**different state schemas**](#different-state-schemas)?
- How to communicate over a [**shared message list**](#shared-message-list)?

#### Graph state vs tool calls

What is the "payload" that is being passed around between agents? In most of the architectures discussed above the agents communicate via the [graph state](#state). In the case of the [supervisor with tool-calling](#supervisor-tool-calling), the payloads are tool call arguments.

![](./img/multi_agent/request.png)

##### Graph state

To communicate via graph state, individual agents need to be defined as [graph nodes](#nodes). These can be added as functions or as entire [subgraphs](#subgraphs). At each step of the graph execution, agent node receives the current state of the graph, executes the agent code and then passes the updated state to the next nodes.

Typically agent nodes share a single [state schema](#schema). However, you might want to design agent nodes with [different state schemas](#different-state-schemas).

#### Different state schemas

An agent might need to have a different state schema from the rest of the agents. For example, a search agent might only need to keep track of queries and retrieved documents. There are two ways to achieve this in LangGraph:

- Define [subgraph](#subgraphs) agents with a separate state schema. If there are no shared state keys (channels) between the subgraph and the parent graph, itÔÇÖs important to [add input / output transformations](https://langchain-ai.github.io/langgraph/how-tos/subgraph-transform-state/) so that the parent graph knows how to communicate with the subgraphs.
- Define agent node functions with a [private input state schema](https://langchain-ai.github.io/langgraph/how-tos/pass_private_state/) that is distinct from the overall graph state schema. This allows passing information that is only needed for executing that particular agent.

#### Shared message list

The most common way for the agents to communicate is via a shared state channel, typically a list of messages. This assumes that there is always at least a single channel (key) in the state that is shared by the agents. When communicating via a shared message list there is an additional consideration: should the agents [share the full history](#share-full-history) of their thought process or only [the final result](#share-final-result)?

![](./img/multi_agent/response.png)

##### Share full history

Agents can **share the full history** of their thought process (i.e. "scratchpad") with all other agents. This "scratchpad" would typically look like a [list of messages](#why-use-messages). The benefit of sharing full thought process is that it might help other agents make better decisions and improve reasoning ability for the system as a whole. The downside is that as the number of agents and their complexity grows, the "scratchpad" will grow quickly and might require additional strategies for [memory management](#managing-long-conversation-history).

##### Share final result

Agents can have their own private "scratchpad" and only **share the final result** with the rest of the agents. This approach might work better for systems with many agents or agents that are more complex. In this case, you would need to define agents with [different state schemas](#different-state-schemas)

For agents called as tools, the supervisor determines the inputs based on the tool schema. Additionally, LangGraph allows [passing state](https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#pass-graph-state-to-tools) to individual tools at runtime, so subordinate agents can access parent state, if needed.


<a name="docsconceptspersistencemd"></a>

## Persistence

LangGraph has a built-in persistence layer, implemented through checkpointers. When you compile graph with a checkpointer, the checkpointer saves a `checkpoint` of the graph state at every super-step. Those checkpoints are saved to a `thread`, which can be accessed after graph execution. Because `threads` allow access to graph's state after execution, several powerful capabilities including human-in-the-loop, memory, time travel, and fault-tolerance are all possible. See [this how-to guide](../how-tos/persistence.ipynb) for an end-to-end example on how to add and use checkpointers with your graph. Below, we'll discuss each of these concepts in more detail. 

![Checkpoints](img/persistence/checkpoints.jpg)

### Threads

A thread is a unique ID or [thread identifier](#threads) assigned to each checkpoint saved by a checkpointer. When invoking graph with a checkpointer, you **must** specify a `thread_id` as part of the `configurable` portion of the config:

```python
{"configurable": {"thread_id": "1"}}
```

### Checkpoints

Checkpoint is a snapshot of the graph state saved at each super-step and is represented by `StateSnapshot` object with the following key properties:

- `config`: Config associated with this checkpoint. 
- `metadata`: Metadata associated with this checkpoint.
- `values`: Values of the state channels at this point in time.
- `next` A tuple of the node names to execute next in the graph.
- `tasks`: A tuple of `PregelTask` objects that contain information about next tasks to be executed. If the step was previously attempted, it will include error information. If a graph was interrupted [dynamically](../how-tos/human_in_the_loop/dynamic_breakpoints.ipynb) from within a node, tasks will contain additional data associated with interrupts.

Let's see what checkpoints are saved when a simple graph is invoked as follows:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)
```

After we run the graph, we expect to see exactly 4 checkpoints:

* empty checkpoint with `START` as the next node to be executed
* checkpoint with the user input `{'foo': '', 'bar': []}` and `node_a` as the next node to be executed
* checkpoint with the outputs of `node_a` `{'foo': 'a', 'bar': ['a']}` and `node_b` as the next node to be executed
* checkpoint with the outputs of `node_b` `{'foo': 'b', 'bar': ['a', 'b']}` and no next nodes to be executed

Note that we `bar` channel values contain outputs from both nodes as we have a reducer for `bar` channel.

#### Get state

When interacting with the saved graph state, you **must** specify a [thread identifier](#threads). You can view the *latest* state of the graph by calling `graph.get_state(config)`. This will return a `StateSnapshot` object that corresponds to the latest checkpoint associated with the thread ID provided in the config or a checkpoint associated with a checkpoint ID for the thread, if provided.

```python
## get the latest state snapshot
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

## get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)
```

In our example, the output of `get_state` will look like this:

```
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']},
    next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
    metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
    created_at='2024-08-29T19:19:38.821749+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}}, tasks=()
)
```

#### Get state history

You can get the full history of the graph execution for a given thread by calling `graph.get_state_history(config)`. This will return a list of `StateSnapshot` objects associated with the thread ID provided in the config. Importantly, the checkpoints will be ordered chronologically with the most recent checkpoint / `StateSnapshot` being the first in the list.

```python
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
```

In our example, the output of `get_state_history` will look like this:

```
[
    StateSnapshot(
        values={'foo': 'b', 'bar': ['a', 'b']},
        next=(),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
        metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
        created_at='2024-08-29T19:19:38.821749+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        tasks=(),
    ),
    StateSnapshot(
        values={'foo': 'a', 'bar': ['a']}, next=('node_b',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        metadata={'source': 'loop', 'writes': {'node_a': {'foo': 'a', 'bar': ['a']}}, 'step': 1},
        created_at='2024-08-29T19:19:38.819946+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        tasks=(PregelTask(id='6fb7314f-f114-5413-a1f3-d37dfe98ff44', name='node_b', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'foo': '', 'bar': []},
        next=('node_a',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        metadata={'source': 'loop', 'writes': None, 'step': 0},
        created_at='2024-08-29T19:19:38.817813+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        tasks=(PregelTask(id='f1b14528-5ee5-579c-949b-23ef9bfbed58', name='node_a', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'bar': []},
        next=('__start__',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        metadata={'source': 'input', 'writes': {'foo': ''}, 'step': -1},
        created_at='2024-08-29T19:19:38.816205+00:00',
        parent_config=None,
        tasks=(PregelTask(id='6d27aa2e-d72b-5504-a36f-8620e54a76dd', name='__start__', error=None, interrupts=()),),
    )
]
```

![State](img/persistence/get_state.jpg)

#### Replay

It's also possible to play-back a prior graph execution. If we `invoking` a graph with a `thread_id` and a `checkpoint_id`, then we will *re-play* the graph from a checkpoint that corresponds to the `checkpoint_id`.

* `thread_id` is simply the ID of a thread. This is always required.
* `checkpoint_id` This identifier refers to a specific checkpoint within a thread. 

You must pass these when invoking the graph as part of the `configurable` portion of the config:

```python
## {"configurable": {"thread_id": "1"}}  # valid config
## {"configurable": {"thread_id": "1", "checkpoint_id": "0c62ca34-ac19-445d-bbb0-5b4984975b2a"}}  # also valid config

config = {"configurable": {"thread_id": "1"}}
graph.invoke(None, config=config)
```

Importantly, LangGraph knows whether a particular checkpoint has been executed previously. If it has, LangGraph simply *re-plays* that particular step in the graph and does not re-execute the step. See this [how to guide on time-travel to learn more about replaying](../how-tos/human_in_the_loop/time-travel.ipynb).

![Replay](img/persistence/re_play.jpg)

#### Update state

In addition to re-playing the graph from specific `checkpoints`, we can also *edit* the graph state. We do this using `graph.update_state()`. This method accepts three different arguments:

##### `config`

The config should contain `thread_id` specifying which thread to update. When only the `thread_id` is passed, we update (or fork) the current state. Optionally, if we include `checkpoint_id` field, then we fork that selected checkpoint.

##### `values`

These are the values that will be used to update the state. Note that this update is treated exactly as any update from a node is treated. This means that these values will be passed to the [reducer](#reducers) functions, if they are defined for some of the channels in the graph state. This means that `update_state` does NOT automatically overwrite the channel values for every channel, but only for the channels without reducers. Let's walk through an example.

Let's assume you have defined the state of your graph with the following schema (see full example above):

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

Let's now assume the current state of the graph is

```
{"foo": 1, "bar": ["a"]}
```

If you update the state as below:

```
graph.update_state(config, {"foo": 2, "bar": ["b"]})
```

Then the new state of the graph will be:

```
{"foo": 2, "bar": ["a", "b"]}
```

The `foo` key (channel) is completely changed (because there is no reducer specified for that channel, so `update_state` overwrites it). However, there is a reducer specified for the `bar` key, and so it appends `"b"` to the state of `bar`.

##### `as_node`

The final thing you can optionally specify when calling `update_state` is `as_node`. If you provided it, the update will be applied as if it came from node `as_node`. If `as_node` is not provided, it will be set to the last node that updated the state, if not ambiguous. The reason this matters is that the next steps to execute depend on the last node to have given an update, so this can be used to control which node executes next. See this [how to guide on time-travel to learn more about forking state](../how-tos/human_in_the_loop/time-travel.ipynb).

![Update](img/persistence/checkpoints_full_story.jpg)

### Memory Store

![Model of shared state](img/persistence/shared_state.png)

A [state schema](#schema) specifies a set of keys that are populated as a graph is executed. As discussed above, state can be written by a checkpointer to a thread at each graph step, enabling state persistence.

But, what if we want to retain some information *across threads*? Consider the case of a chatbot where we want to retain specific information about the user across *all* chat conversations (e.g., threads) with that user!

With checkpointers alone, we cannot share information across threads. This motivates the need for the [`Store`](#langgraph.store.base.BaseStore) interface. As an illustration, we can define an `InMemoryStore` to store information about a user across threads. We simply compile our graph with a checkpointer, as before, and with our new `in_memory_store` variable.

#### Basic Usage

First, let's showcase this in isolation without using LangGraph.

```python
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
```

Memories are namespaced by a `tuple`, which in this specific example will be `(<user_id>, "memories")`. The namespace can be any length and represent anything, does not have be user specific.

```python 
user_id = "1"
namespace_for_memory = (user_id, "memories")
```

We use the `store.put` method to save memories to our namespace in the store. When we do this, we specify the namespace, as defined above, and a key-value pair for the memory: the key is simply a unique identifier for the memory (`memory_id`) and the value (a dictionary) is the memory itself.

```python
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)
```

We can read out memories in our namespace using the `store.search` method, which will return all memories for a given user as a list. The most recent memory is the last in the list.

```python
memories = in_memory_store.search(namespace_for_memory)
memories[-1].dict()
{'value': {'food_preference': 'I like pizza'},
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
 'namespace': ['1', 'memories'],
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

Each memory type is a Python class ([`Item`](https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.Item)) with certain attributes. We can access it as a dictionary by converting via `.dict` as above.
The attributes it has are:

- `value`: The value (itself a dictionary) of this memory
- `key`: A unique key for this memory in this namespace
- `namespace`: A list of strings, the namespace of this memory type
- `created_at`: Timestamp for when this memory was created
- `updated_at`: Timestamp for when this memory was updated

#### Semantic Search

Beyond simple retrieval, the store also supports semantic search, allowing you to find memories based on meaning rather than exact matches. To enable this, configure the store with an embedding model:

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
        "dims": 1536,                              # Embedding dimensions
        "fields": ["food_preference", "$"]              # Fields to embed
    }
)
```

Now when searching, you can use natural language queries to find relevant memories:

```python
## Find memories about food preferences
## (This can be done after putting memories into the store)
memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3  # Return top 3 matches
)
```

You can control which parts of your memories get embedded by configuring the `fields` parameter or by specifying the `index` parameter when storing memories:

```python
## Store with specific fields to embed
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans"
    },
    index=["food_preference"]  # Only embed "food_preferences" field
)

## Store without embedding (still retrievable, but not searchable)
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)
```

#### Using in LangGraph

With this all in place, we use the `in_memory_store` in LangGraph. The `in_memory_store` works hand-in-hand with the checkpointer: the checkpointer saves state to threads, as discussed above, and the `in_memory_store` allows us to store arbitrary information for access *across* threads. We compile the graph with both the checkpointer and the `in_memory_store` as follows. 

```python
from langgraph.checkpoint.memory import MemorySaver

## We need this because we want to enable threads (conversations)
checkpointer = MemorySaver()

## ... Define the graph ...

## Compile the graph with the checkpointer and store
graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
```

We invoke the graph with a `thread_id`, as before, and also with a `user_id`, which we'll use to namespace our memories to this particular user as we showed above.

```python
## Invoke the graph
user_id = "1"
config = {"configurable": {"thread_id": "1", "user_id": user_id}}

## First let's just say hi to the AI
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]}, config, stream_mode="updates"
):
    print(update)
```

We can access the `in_memory_store` and the `user_id` in *any node* by passing `store: BaseStore` and `config: RunnableConfig` as node arguments. Here's how we might use semantic search in a node to find relevant memories:

```python
def update_memory(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    
    # Get the user id from the config
    user_id = config["configurable"]["user_id"]
    
    # Namespace the memory
    namespace = (user_id, "memories")
    
    # ... Analyze conversation and create a new memory
    
    # Create a new memory ID
    memory_id = str(uuid.uuid4())

    # We create a new memory
    store.put(namespace, memory_id, {"memory": memory})

```

As we showed above, we can also access the store in any node and use the `store.search` method to get memories. Recall the the memories are returned as a list of objects that can be converted to a dictionary.

```python
memories[-1].dict()
{'value': {'food_preference': 'I like pizza'},
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
 'namespace': ['1', 'memories'],
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

We can access the memories and use them in our model call.

```python
def call_model(state: MessagesState, config: RunnableConfig, *, store: BaseStore):
    # Get the user id from the config
    user_id = config["configurable"]["user_id"]
    
    # Search based on the most recent message
    memories = store.search(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])
    
    # ... Use memories in the model call
```

If we create a new thread, we can still access the same memories so long as the `user_id` is the same. 

```python
## Invoke the graph
config = {"configurable": {"thread_id": "2", "user_id": "1"}}

## Let's say hi again
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]}, config, stream_mode="updates"
):
    print(update)
```

When we use the LangGraph Platform, either locally (e.g., in LangGraph Studio) or with LangGraph Cloud, the base store is available to use by default and does not need to be specified during graph compilation. To enable semantic search, however, you **do** need to configure the indexing settings in your `langgraph.json` file. For example:

```json
{
    ...
    "store": {
        "index": {
            "embed": "openai:text-embeddings-3-small",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

See the [deployment guide](#docsclouddeploymentsemantic_searchmd) for more details and configuration options.

### Checkpointer libraries

Under the hood, checkpointing is powered by checkpointer objects that conform to [BaseCheckpointSaver][langgraph.checkpoint.base.BaseCheckpointSaver] interface. LangGraph provides several checkpointer implementations, all implemented via standalone, installable libraries:

* `langgraph-checkpoint`: The base interface for checkpointer savers ([BaseCheckpointSaver][langgraph.checkpoint.base.BaseCheckpointSaver]) and serialization/deserialization interface ([SerializerProtocol][langgraph.checkpoint.serde.base.SerializerProtocol]). Includes in-memory checkpointer implementation ([MemorySaver][langgraph.checkpoint.memory.MemorySaver]) for experimentation. LangGraph comes with `langgraph-checkpoint` included.
* `langgraph-checkpoint-sqlite`: An implementation of LangGraph checkpointer that uses SQLite database ([SqliteSaver][langgraph.checkpoint.sqlite.SqliteSaver] / [AsyncSqliteSaver][langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver]). Ideal for experimentation and local workflows. Needs to be installed separately.
* `langgraph-checkpoint-postgres`: An advanced checkpointer that uses Postgres database ([PostgresSaver][langgraph.checkpoint.postgres.PostgresSaver] / [AsyncPostgresSaver][langgraph.checkpoint.postgres.aio.AsyncPostgresSaver]), used in LangGraph Cloud. Ideal for using in production. Needs to be installed separately.

#### Checkpointer interface

Each checkpointer conforms to [BaseCheckpointSaver][langgraph.checkpoint.base.BaseCheckpointSaver] interface and implements the following methods:

* `.put` - Store a checkpoint with its configuration and metadata.  
* `.put_writes` - Store intermediate writes linked to a checkpoint (i.e. [pending writes](#pending-writes)).  
* `.get_tuple` - Fetch a checkpoint tuple using for a given configuration (`thread_id` and `checkpoint_id`). This is used to populate `StateSnapshot` in `graph.get_state()`.  
* `.list` - List checkpoints that match a given configuration and filter criteria. This is used to populate state history in `graph.get_state_history()`

If the checkpointer is used with asynchronous graph execution (i.e. executing the graph via `.ainvoke`, `.astream`, `.abatch`), asynchronous versions of the above methods will be used (`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`).

!!! note Note
    For running your graph asynchronously, you can use `MemorySaver`, or async versions of Sqlite/Postgres checkpointers -- `AsyncSqliteSaver` / `AsyncPostgresSaver` checkpointers.

#### Serializer

When checkpointers save the graph state, they need to serialize the channel values in the state. This is done using serializer objects. 
`langgraph_checkpoint` defines [protocol][langgraph.checkpoint.serde.base.SerializerProtocol] for implementing serializers provides a default implementation ([JsonPlusSerializer][langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer]) that handles a wide variety of types, including LangChain and LangGraph primitives, datetimes, enums and more.

### Capabilities

#### Human-in-the-loop

First, checkpointers facilitate [human-in-the-loop workflows](#human-in-the-loop) workflows by allowing humans to inspect, interrupt, and approve graph steps. Checkpointers are needed for these workflows as the human has to be able to view the state of a graph at any point in time, and the graph has to be to resume execution after the human has made any updates to the state. See [these how-to guides](../how-tos/human_in_the_loop/breakpoints.ipynb) for concrete examples.

#### Memory

Second, checkpointers allow for ["memory"](#memory) between interactions.  In the case of repeated human interactions (like conversations) any follow up messages can be sent to that thread, which will retain its memory of previous ones. See [this how-to guide](../how-tos/memory/manage-conversation-history.ipynb) for an end-to-end example on how to add and manage conversation memory using checkpointers.

#### Time Travel

Third, checkpointers allow for ["time travel"](#docsconceptstime-travelmd), allowing users to replay prior graph executions to review and / or debug specific graph steps. In addition, checkpointers make it possible to fork the graph state at arbitrary checkpoints to explore alternative trajectories.

#### Fault-tolerance

Lastly, checkpointing also provides fault-tolerance and error recovery: if one or more nodes fail at a given superstep, you can restart your graph from the last successful step. Additionally, when a graph node fails mid-execution at a given superstep, LangGraph stores pending checkpoint writes from any other nodes that completed successfully at that superstep, so that whenever we resume graph execution from that superstep we don't re-run the successful nodes.

##### Pending writes

Additionally, when a graph node fails mid-execution at a given superstep, LangGraph stores pending checkpoint writes from any other nodes that completed successfully at that superstep, so that whenever we resume graph execution from that superstep we don't re-run the successful nodes.


<a name="docsconceptsplansmd"></a>

## LangGraph Platform Plans


### Overview
LangGraph Platform is a commercial solution for deploying agentic applications in production.
There are three different plans for using it.

- **Developer**: All [LangSmith](https://smith.langchain.com/) users have access to this plan. You can sign up for this plan simply by creating a LangSmith account. This gives you access to the [Self-Hosted Lite](#self-hosted-lite) deployment option.
- **Plus**: All [LangSmith](https://smith.langchain.com/) users with a [Plus account](https://docs.smith.langchain.com/administration/pricing) have access to this plan. You can sign up for this plan simply by upgrading your LangSmith account to the Plus plan type. This gives you access to the [Cloud](#cloud-saas) deployment option.
- **Enterprise**: This is separate from LangSmith plans. You can sign up for this plan by contacting sales@langchain.dev. This gives you access to all deployment options: [Cloud](#cloud-saas), [Bring-Your-Own-Cloud](#bring-your-own-cloud), and [Self Hosted Enterprise](#self-hosted-enterprise)


### Plan Details

|                                                                  | Developer                                   | Plus                                                  | Enterprise                                          |
|------------------------------------------------------------------|---------------------------------------------|-------------------------------------------------------|-----------------------------------------------------|
| Deployment Options                                               | Self-Hosted Lite                            | Cloud                                                 | Self-Hosted Enterprise, Cloud, Bring-Your-Own-Cloud |
| Usage                                                     | Free, limited to 1M nodes executed per year | Free while in Beta, will be charged per node executed | Custom                                              |
| APIs for retrieving and updating state and conversational history | Ô£à                                           | Ô£à                                                     | Ô£à                                                   |
| APIs for retrieving and updating long-term memory                | Ô£à                                           | Ô£à                                                     | Ô£à                                                   |
| Horizontally scalable task queues and servers                    | Ô£à                                           | Ô£à                                                     | Ô£à                                                   |
| Real-time streaming of outputs and intermediate steps            | Ô£à                                           | Ô£à                                                     | Ô£à                                                   |
| Assistants API (configurable templates for LangGraph apps)       | Ô£à                                           | Ô£à                                                     | Ô£à                                                   |
| Cron scheduling                                                  | --                                          | Ô£à                                                     | Ô£à                                                   |
| LangGraph Studio for prototyping                                 | Desktop only                                | Coming Soon!                                          | Coming Soon!                                        |
| Authentication & authorization to call the LangGraph APIs        | --                                          | Coming Soon!                                          | Coming Soon!                                        |
| Smart caching to reduce traffic to LLM API                       | --                                          | Coming Soon!                                          | Coming Soon!                                        |
| Publish/subscribe API for state                                  | --                                          | Coming Soon!                                          | Coming Soon!                                        |
| Scheduling prioritization                                        | --                                          | Coming Soon!                                          | Coming Soon!                                        |

Please see the [LangGraph Platform Pricing](https://www.langchain.com/langgraph-platform-pricing) for information on pricing.

### Related

For more information, please see:

* [Deployment Options conceptual guide](#docsconceptsdeployment_optionsmd)
* [LangGraph Platform Pricing](https://www.langchain.com/langgraph-platform-pricing)
* [LangSmith Plans](https://docs.smith.langchain.com/administration/pricing)


<a name="docsconceptssdkmd"></a>

## LangGraph SDK

!!! info "Prerequisites"
    - [LangGraph Platform](#docsconceptslanggraph_platformmd)
    - [LangGraph Server](#docsconceptslanggraph_servermd)

The LangGraph Platform provides both a Python and JS SDK for interacting with the [LangGraph Server API](#docsconceptslanggraph_servermd). 

### Installation

You can install the packages using the appropriate package manager for your language.

=== "Python"
    ```bash
    pip install langgraph-sdk
    ```

=== "JS"
    ```bash
    yarn add @langchain/langgraph-sdk
    ```


### API Reference

You can find the API reference for the SDKs here:

- [Python SDK Reference](#docscloudreferencesdkpython_sdk_refmd)
- [JS/TS SDK Reference](../cloud/reference/sdk/js_ts_sdk_ref.md)

### Python Sync vs. Async

The Python SDK provides both synchronous (`get_sync_client`) and asynchronous (`get_client`) clients for interacting with the LangGraph Server API.

=== "Async"
    ```python
    from langgraph_sdk import get_client

    client = get_client(url=..., api_key=...)
    await client.assistants.search()
    ```

=== "Sync"

    ```python
    from langgraph_sdk import get_sync_client

    client = get_sync_client(url=..., api_key=...)
    client.assistants.search()
    ```

### Related

- [LangGraph CLI API Reference](#docscloudreferenceclimd)
- [Python SDK Reference](#docscloudreferencesdkpython_sdk_refmd)
- [JS/TS SDK Reference](../cloud/reference/sdk/js_ts_sdk_ref.md)

<a name="docsconceptsself_hostedmd"></a>

## Self-Hosted

!!! note Prerequisites

    - [LangGraph Platform](#docsconceptslanggraph_platformmd)
    - [Deployment Options](#docsconceptsdeployment_optionsmd)

### Versions

There are two versions of the self-hosted deployment: [Self-Hosted Enterprise](#self-hosted-enterprise) and [Self-Hosted Lite](#self-hosted-lite).

#### Self-Hosted Lite

The Self-Hosted Lite version is a limited version of LangGraph Platform that you can run locally or in a self-hosted manner (up to 1 million nodes executed).

When using the Self-Hosted Lite version, you authenticate with a [LangSmith](https://smith.langchain.com/) API key.

#### Self-Hosted Enterprise

The Self-Hosted Enterprise version is the full version of LangGraph Platform.

To use the Self-Hosted Enterprise version, you must acquire a license key that you will need to pass in when running the Docker image. To acquire a license key, please email sales@langchain.dev.

### Requirements

- You use `langgraph-cli` and/or [LangGraph Studio](#docsconceptslanggraph_studiomd) app to test graph locally.
- You use `langgraph build` command to build image.

### How it works

- Deploy Redis and Postgres instances on your own infrastructure.
- Build the docker image for [LangGraph Server](#docsconceptslanggraph_servermd) using the [LangGraph CLI](#docsconceptslanggraph_climd).
- Deploy a web server that will run the docker image and pass in the necessary environment variables.

For step-by-step instructions, see [How to set up a self-hosted deployment of LangGraph](#docshow-tosdeploy-self-hostedmd).

### Helm Chart

If you would like to deploy LangGraph Cloud on Kubernetes, you can use this [Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md).

### Related

- [How to set up a self-hosted deployment of LangGraph](#docshow-tosdeploy-self-hostedmd).


<a name="docsconceptsstreamingmd"></a>

## Streaming

LangGraph is built with first class support for streaming. There are several different ways to stream back outputs from a graph run

### Streaming graph outputs (`.stream` and `.astream`)

`.stream` and `.astream` are sync and async methods for streaming back outputs from a graph run.
There are several different modes you can specify when calling these methods (e.g. `graph.stream(..., mode="...")):

- [`"values"`](../how-tos/stream-values.ipynb): This streams the full value of the state after each step of the graph.
- [`"updates"`](../how-tos/stream-updates.ipynb): This streams the updates to the state after each step of the graph. If multiple updates are made in the same step (e.g. multiple nodes are run) then those updates are streamed separately.
- [`"custom"`](../how-tos/streaming-content.ipynb): This streams custom data from inside your graph nodes.
- [`"messages"`](../how-tos/streaming-tokens.ipynb): This streams LLM tokens and metadata for the graph node where LLM is invoked.
- `"debug"`: This streams as much information as possible throughout the execution of the graph.

You can also specify multiple streaming modes at the same time by passing them as a list. When you do this, the streamed outputs will be tuples `(stream_mode, data)`. For example:

```python
graph.stream(..., stream_mode=["updates", "messages"])
```

```
...
('messages', (AIMessageChunk(content='Hi'), {'langgraph_step': 3, 'langgraph_node': 'agent', ...}))
...
('updates', {'agent': {'messages': [AIMessage(content="Hi, how can I help you?")]}})
```

The below visualization shows the difference between the `values` and `updates` modes:

![values vs updates](../static/values_vs_updates.png)


### Streaming LLM tokens and events (`.astream_events`)

In addition, you can use the [`astream_events`](../how-tos/streaming-events-from-within-tools.ipynb) method to stream back events that happen _inside_ nodes. This is useful for [streaming tokens of LLM calls](../how-tos/streaming-tokens.ipynb).

This is a standard method on all [LangChain objects](https://python.langchain.com/docs/concepts/#runnable-interface). This means that as the graph is executed, certain events are emitted along the way and can be seen if you run the graph using `.astream_events`. 

All events have (among other things) `event`, `name`, and `data` fields. What do these mean?

- `event`: This is the type of event that is being emitted. You can find a detailed table of all callback events and triggers [here](https://python.langchain.com/docs/concepts/#callback-events).
- `name`: This is the name of event.
- `data`: This is the data associated with the event.

What types of things cause events to be emitted?

* each node (runnable) emits `on_chain_start` when it starts execution, `on_chain_stream` during the node execution and `on_chain_end` when the node finishes. Node events will have the node name in the event's `name` field
* the graph will emit `on_chain_start` in the beginning of the graph execution, `on_chain_stream` after each node execution and `on_chain_end` when the graph finishes. Graph events will have the `LangGraph` in the event's `name` field
* Any writes to state channels (i.e. anytime you update the value of one of your state keys) will emit `on_chain_start` and `on_chain_end` events

Additionally, any events that are created inside your nodes (LLM events, tool events, manually emitted events, etc.) will also be visible in the output of `.astream_events`.

To make this more concrete and to see what this looks like, let's see what events are returned when we run a simple graph:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI(model="gpt-4o-mini")


def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node(call_model)
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)
app = workflow.compile()

inputs = [{"role": "user", "content": "hi!"}]
async for event in app.astream_events({"messages": inputs}, version="v1"):
    kind = event["event"]
    print(f"{kind}: {event['name']}")
```
```shell
on_chain_start: LangGraph
on_chain_start: __start__
on_chain_end: __start__
on_chain_start: call_model
on_chat_model_start: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_end: ChatOpenAI
on_chain_start: ChannelWrite<call_model,messages>
on_chain_end: ChannelWrite<call_model,messages>
on_chain_stream: call_model
on_chain_end: call_model
on_chain_stream: LangGraph
on_chain_end: LangGraph
```

We start with the overall graph start (`on_chain_start: LangGraph`). We then write to the `__start__` node (this is special node to handle input).
We then start the `call_model` node (`on_chain_start: call_model`). We then start the chat model invocation (`on_chat_model_start: ChatOpenAI`),
stream back token by token (`on_chat_model_stream: ChatOpenAI`) and then finish the chat model (`on_chat_model_end: ChatOpenAI`). From there, 
we write the results back to the channel (`ChannelWrite<call_model,messages>`) and then finish the `call_model` node and then the graph as a whole.

This should hopefully give you a good sense of what events are emitted in a simple graph. But what data do these events contain?
Each type of event contains data in a different format. Let's look at what `on_chat_model_stream` events look like. This is an important type of event
since it is needed for streaming tokens from an LLM response.

These events look like:

```shell
{'event': 'on_chat_model_stream',
 'name': 'ChatOpenAI',
 'run_id': '3fdbf494-acce-402e-9b50-4eab46403859',
 'tags': ['seq:step:1'],
 'metadata': {'langgraph_step': 1,
  'langgraph_node': 'call_model',
  'langgraph_triggers': ['start:call_model'],
  'langgraph_task_idx': 0,
  'checkpoint_id': '1ef657a0-0f9d-61b8-bffe-0c39e4f9ad6c',
  'checkpoint_ns': 'call_model',
  'ls_provider': 'openai',
  'ls_model_name': 'gpt-4o-mini',
  'ls_model_type': 'chat',
  'ls_temperature': 0.7},
 'data': {'chunk': AIMessageChunk(content='Hello', id='run-3fdbf494-acce-402e-9b50-4eab46403859')},
 'parent_ids': []}
```
We can see that we have the event type and name (which we knew from before).

We also have a bunch of stuff in metadata. Noticeably, `'langgraph_node': 'call_model',` is some really helpful information
which tells us which node this model was invoked inside of.

Finally, `data` is a really important field. This contains the actual data for this event! Which in this case
is an AIMessageChunk. This contains the `content` for the message, as well as an `id`.
This is the ID of the overall AIMessage (not just this chunk) and is super helpful - it helps
us track which chunks are part of the same message (so we can show them together in the UI).

This information contains all that is needed for creating a UI for streaming LLM tokens. You can see a 
guide for that [here](../how-tos/streaming-tokens.ipynb).


!!! warning "ASYNC IN PYTHON<=3.10"
    You may fail to see events being emitted from inside a node when using `.astream_events` in Python <= 3.10. If you're using a Langchain RunnableLambda, a RunnableGenerator, or Tool asynchronously inside your node, you will have to propagate callbacks to these objects manually. This is because LangChain cannot automatically propagate callbacks to child objects in this case. Please see examples [here](../how-tos/streaming-content.ipynb) and [here](../how-tos/streaming-events-from-within-tools.ipynb).


### LangGraph Platform

Streaming is critical for making LLM applications feel responsive to end users. When creating a streaming run, the streaming mode determines what data is streamed back to the API client. LangGraph Platform supports five streaming modes:

- `values`: Stream the full state of the graph after each [super-step](https://langchain-ai.github.io/langgraph/concepts/low_level/#graphs) is executed. See the [how-to guide](#docscloudhow-tosstream_valuesmd) for streaming values.
- `messages-tuple`: Stream LLM tokens for any messages generated inside a node. This mode is primarily meant for powering chat applications. See the [how-to guide](#docscloudhow-tosstream_messagesmd) for streaming messages.
- `updates`: Streams updates to the state of the graph after each node is executed. See the [how-to guide](#docscloudhow-tosstream_updatesmd) for streaming updates.
- `events`: Stream all events (including the state of the graph) that occur during graph execution. See the [how-to guide](#docscloudhow-tosstream_eventsmd) for streaming events. This can be used to do token-by-token streaming for LLMs.
- `debug`: Stream debug events throughout graph execution. See the [how-to guide](#docscloudhow-tosstream_debugmd) for streaming debug events.

You can also specify multiple streaming modes at the same time. See the [how-to guide](#docscloudhow-tosstream_multiplemd) for configuring multiple streaming modes at the same time.

See the [API reference](#tag\threads-runs\POST\threads\{thread_id}\runs\stream) for how to create streaming runs.

Streaming modes `values`, `updates`, `messages-tuple` and `debug` are very similar to modes available in the LangGraph library - for a deeper conceptual explanation of those, you can see the [previous section](#streaming-graph-outputs-stream-and-astream).

Streaming mode `events` is the same as using `.astream_events` in the LangGraph library - for a deeper conceptual explanation of this, you can see the [previous section](#streaming-graph-outputs-stream-and-astream).

All events emitted have two attributes:

- `event`: This is the name of the event
- `data`: This is data associated with the event

<a name="docsconceptstemplate_applicationsmd"></a>

## Template Applications

Templates are open source reference applications designed to help you get started quickly when building with LangGraph. They provide working examples of common agentic workflows that can be customized to your needs.

You can create an application from a template using the LangGraph CLI.

!!! info "Requirements"

    - Python >= 3.11
    - [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/): Requires langchain-cli[inmem] >= 0.1.58

### Install the LangGraph CLI

```bash
pip install "langgraph-cli[inmem]" --upgrade
```

### Available Templates

| Template                  | Description                                                                              | Python                                                           | JS/TS                                                               |
|---------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------|---------------------------------------------------------------------|
| **New LangGraph Project** | A simple, minimal chatbot with memory.                                                   | [Repo](https://github.com/langchain-ai/new-langgraph-project)    | [Repo](https://github.com/langchain-ai/new-langgraphjs-project)     |
| **ReAct Agent**           | A simple agent that can be flexibly extended to many tools.                              | [Repo](https://github.com/langchain-ai/react-agent)              | [Repo](https://github.com/langchain-ai/react-agent-js)              |
| **Memory Agent**          | A ReAct-style agent with an additional tool to store memories for use across threads.    | [Repo](https://github.com/langchain-ai/memory-agent)             | [Repo](https://github.com/langchain-ai/memory-agent-js)             |
| **Retrieval Agent**       | An agent that includes a retrieval-based question-answering system.                      | [Repo](https://github.com/langchain-ai/retrieval-agent-template) | [Repo](https://github.com/langchain-ai/retrieval-agent-template-js) |
| **Data-Enrichment Agent** | An agent that performs web searches and organizes its findings into a structured format. | [Repo](https://github.com/langchain-ai/data-enrichment)          | [Repo](https://github.com/langchain-ai/data-enrichment-js)          |


### ­ƒî▒ Create a LangGraph App

To create a new app from a template, use the `langgraph new` command.

```bash
langgraph new
```

### Next Steps

Review the `README.md` file in the root of your new LangGraph app for more information about the template and how to customize it.

After configuring the app properly and adding your API keys, you can start the app using the LangGraph CLI:

```bash
langgraph dev 
```

See the following guides for more information on how to deploy your app:

- **[Launch Local LangGraph Server](#docstutorialslanggraph-platformlocal-servermd)**: This quick start guide shows how to start a LangGraph Server locally for the **ReAct Agent** template. The steps are similar for other templates.
- **[Deploy to LangGraph Cloud](#docscloudquick_startmd)**: Deploy your LangGraph app using LangGraph Cloud.
 
#### LangGraph Framework

- **[LangGraph Concepts](#docsconceptsindexmd)**: Learn the foundational concepts of LangGraph.
- **[LangGraph How-to Guides](#docshow-tosindexmd)**: Guides for common tasks with LangGraph.

#### ­ƒôÜ Learn More about LangGraph Platform

Expand your knowledge with these resources:

- **[LangGraph Platform Concepts](#langgraph-platform)**: Understand the foundational concepts of the LangGraph Platform.
- **[LangGraph Platform How-to Guides](#langgraph-platform)**: Discover step-by-step guides to build and deploy applications.


<a name="docsconceptstime-travelmd"></a>

## Time Travel ÔÅ▒´©Å

!!! note "Prerequisites"

    This guide assumes that you are familiar with LangGraph's checkpoints and states. If not, please review the [persistence](#docsconceptspersistencemd) concept first.


When working with non-deterministic systems that make model-based decisions (e.g., agents powered by LLMs), it can be useful to examine their decision-making process in detail:

1. ­ƒñö **Understand Reasoning**: Analyze the steps that led to a successful result.
2. ­ƒÉ× **Debug Mistakes**: Identify where and why errors occurred.
3. ­ƒöì **Explore Alternatives**: Test different paths to uncover better solutions.

We call these debugging techniques **Time Travel**, composed of two key actions: [**Replaying**](#replaying) ­ƒöü and [**Forking**](#forking) ­ƒöÇ .

### Replaying

![](./img/human_in_the_loop/replay.png)

Replaying allows us to revisit and reproduce an agent's past actions. This can be done either from the current state (or checkpoint) of the graph or from a specific checkpoint.

To replay from the current state, simply pass `None` as the input along with a `thread`:

```python
thread = {"configurable": {"thread_id": "1"}}
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

To replay actions from a specific checkpoint, start by retrieving all checkpoints for the thread:

```python
all_checkpoints = []
for state in graph.get_state_history(thread):
    all_checkpoints.append(state)
```

Each checkpoint has a unique ID. After identifying the desired checkpoint, for instance, `xyz`, include its ID in the configuration:

```python
config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz'}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```

The graph efficiently replays previously executed nodes instead of re-executing them, leveraging its awareness of prior checkpoint executions.

### Forking

![](./img/human_in_the_loop/forking.png)

Forking allows you to revisit an agent's past actions and explore alternative paths within the graph.

To edit a specific checkpoint, such as `xyz`, provide its `checkpoint_id` when updating the graph's state:

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "xyz"}}
graph.update_state(config, {"state": "updated state"})
```

This creates a new forked checkpoint, xyz-fork, from which you can continue running the graph:

```python
config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xyz-fork'}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```

### Additional Resources ­ƒôÜ

- [**Conceptual Guide: Persistence**](https://langchain-ai.github.io/langgraph/concepts/persistence/#replay): Read the persistence guide for more context on replaying.
- [**How to View and Update Past Graph State**](../how-tos/human_in_the_loop/time-travel.ipynb): Step-by-step instructions for working with graph state that demonstrate the **replay** and **fork** actions.


<a name="docsconceptsv0-human-in-the-loopmd"></a>

## Human-in-the-loop

!!! note "Use the `interrupt` function instead."

    As of LangGraph 0.2.57, the recommended way to set breakpoints is using the [`interrupt` function][langgraph.types.interrupt] as it simplifies **human-in-the-loop** patterns.

    Please see the revised [human-in-the-loop guide](#docsconceptshuman_in_the_loopmd) for the latest version that uses the `interrupt` function.


Human-in-the-loop (or "on-the-loop") enhances agent capabilities through several common user interaction patterns.

Common interaction patterns include:

(1) `Approval` - We can interrupt our agent, surface the current state to a user, and allow the user to accept an action. 

(2) `Editing` - We can interrupt our agent, surface the current state to a user, and allow the user to edit the agent state. 

(3) `Input` - We can explicitly create a graph node to collect human input and pass that input directly to the agent state.

Use-cases for these interaction patterns include:

(1) `Reviewing tool calls` - We can interrupt an agent to review and edit the results of tool calls.

(2) `Time Travel` - We can manually re-play and / or fork past actions of an agent.

### Persistence

All of these interaction patterns are enabled by LangGraph's built-in [persistence](#docsconceptspersistencemd) layer, which will write a checkpoint of the graph state at each step. Persistence allows the graph to stop so that a human can review and / or edit the current state of the graph and then resume with the human's input.

#### Breakpoints

Adding a [breakpoint](#docsconceptsbreakpointsmd) a specific location in the graph flow is one way to enable human-in-the-loop. In this case, the developer knows *where* in the workflow human input is needed and simply places a breakpoint prior to or following that particular graph node.

Here, we compile our graph with a checkpointer and a breakpoint at the node we want to interrupt before, `step_for_human_in_the_loop`. We then perform one of the above interaction patterns, which will create a new checkpoint if a human edits the graph state. The new checkpoint is saved to the `thread` and we can resume the graph execution from there by passing in `None` as the input.

```python
## Compile our graph with a checkpointer and a breakpoint before "step_for_human_in_the_loop"
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["step_for_human_in_the_loop"])

## Run the graph up to the breakpoint
thread_config = {"configurable": {"thread_id": "1"}}
for event in graph.stream(inputs, thread_config, stream_mode="values"):
    print(event)
    
## Perform some action that requires human in the loop

## Continue the graph execution from the current checkpoint 
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

#### Dynamic Breakpoints

Alternatively, the developer can define some *condition* that must be met for a breakpoint to be triggered. This concept of [dynamic breakpoints](#docsconceptsbreakpointsmd) is useful when the developer wants to halt the graph under *a particular condition*. This uses a `NodeInterrupt`, which is a special type of exception that can be raised from within a node based upon some condition. As an example, we can define a dynamic breakpoint that triggers when the `input` is longer than 5 characters.

```python
def my_node(state: State) -> State:
    if len(state['input']) > 5:
        raise NodeInterrupt(f"Received input that is longer than 5 characters: {state['input']}")
    return state
```

Let's assume we run the graph with an input that triggers the dynamic breakpoint and then attempt to resume the graph execution simply by passing in `None` for the input. 

```python
## Attempt to continue the graph execution with no change to state after we hit the dynamic breakpoint 
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

The graph will *interrupt* again because this node will be *re-run* with the same graph state. We need to change the graph state such that the condition that triggers the dynamic breakpoint is no longer met. So, we can simply edit the graph state to an input that meets the condition of our dynamic breakpoint (< 5 characters) and re-run the node.

```python 
## Update the state to pass the dynamic breakpoint
graph.update_state(config=thread_config, values={"input": "foo"})
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

Alternatively, what if we want to keep our current input and skip the node (`my_node`) that performs the check? To do this, we can simply perform the graph update with `as_node="my_node"` and pass in `None` for the values. This will make no update the graph state, but run the update as `my_node`, effectively skipping the node and bypassing the dynamic breakpoint.

```python
## This update will skip the node `my_node` altogether
graph.update_state(config=thread_config, values=None, as_node="my_node")
for event in graph.stream(None, thread_config, stream_mode="values"):
    print(event)
```

See [our guide](../how-tos/human_in_the_loop/dynamic_breakpoints.ipynb) for a detailed how-to on doing this!

### Interaction Patterns

#### Approval

![](./img/human_in_the_loop/approval.png)

Sometimes we want to approve certain steps in our agent's execution. 
 
We can interrupt our agent at a [breakpoint](#docsconceptsbreakpointsmd) prior to the step that we want to approve.

This is generally recommend for sensitive actions (e.g., using external APIs or writing to a database).
 
With persistence, we can surface the current agent state as well as the next step to a user for review and approval. 
 
If approved, the graph resumes execution from the last saved checkpoint, which is saved to the `thread`:

```python
## Compile our graph with a checkpointer and a breakpoint before the step to approve
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["node_2"])

## Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
## ... Get human approval ...

## If approved, continue the graph execution from the last saved checkpoint
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [our guide](../how-tos/human_in_the_loop/breakpoints.ipynb) for a detailed how-to on doing this!

#### Editing

![](./img/human_in_the_loop/edit_graph_state.png)

Sometimes we want to review and edit the agent's state. 
 
As with approval, we can interrupt our agent at a [breakpoint](#docsconceptsbreakpointsmd) prior to the step we want to check. 
 
We can surface the current state to a user and allow the user to edit the agent state.
 
This can, for example, be used to correct the agent if it made a mistake (e.g., see the section on tool calling below).

We can edit the graph state by forking the current checkpoint, which is saved to the `thread`.

We can then proceed with the graph from our forked checkpoint as done before. 

```python
## Compile our graph with a checkpointer and a breakpoint before the step to review
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["node_2"])

## Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
## Review the state, decide to edit it, and create a forked checkpoint with the new state
graph.update_state(thread, {"state": "new state"})

## Continue the graph execution from the forked checkpoint
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [this guide](../how-tos/human_in_the_loop/edit-graph-state.ipynb) for a detailed how-to on doing this!

#### Input

![](./img/human_in_the_loop/wait_for_input.png)

Sometimes we want to explicitly get human input at a particular step in the graph. 
 
We can create a graph node designated for this (e.g., `human_input` in our example diagram).
 
As with approval and editing, we can interrupt our agent at a [breakpoint](#docsconceptsbreakpointsmd) prior to this node.
 
We can then perform a state update that includes the human input, just as we did with editing state.

But, we add one thing: 

We can use `as_node=human_input` with the state update to specify that the state update *should be treated as a node*.

The is subtle, but important: 

With editing, the user makes a decision about whether or not to edit the graph state.

With input, we explicitly define a node in our graph for collecting human input!

The state update with the human input then runs *as this node*.

```python
## Compile our graph with a checkpointer and a breakpoint before the step to to collect human input
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_input"])

## Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
## Update the state with the user input as if it was the human_input node
graph.update_state(thread, {"user_input": user_input}, as_node="human_input")

## Continue the graph execution from the checkpoint created by the human_input node
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [this guide](../how-tos/human_in_the_loop/wait-user-input.ipynb) for a detailed how-to on doing this!

### Use-cases

#### Reviewing Tool Calls

Some user interaction patterns combine the above ideas.

For example, many agents use [tool calling](https://python.langchain.com/docs/how_to/tool_calling/) to make decisions. 

Tool calling presents a challenge because the agent must get two things right: 

(1) The name of the tool to call 

(2) The arguments to pass to the tool

Even if the tool call is correct, we may also want to apply discretion: 

(3) The tool call may be a sensitive operation that we want to approve 

With these points in mind, we can combine the above ideas to create a human-in-the-loop review of a tool call.

```python
## Compile our graph with a checkpointer and a breakpoint before the step to to review the tool call from the LLM 
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

## Run the graph up to the breakpoint
for event in graph.stream(inputs, thread, stream_mode="values"):
    print(event)
    
## Review the tool call and update it, if needed, as the human_review node
graph.update_state(thread, {"tool_call": "updated tool call"}, as_node="human_review")

## Otherwise, approve the tool call and proceed with the graph execution with no edits 

## Continue the graph execution from either: 
## (1) the forked checkpoint created by human_review or 
## (2) the checkpoint saved when the tool call was originally made (no edits in human_review)
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

See [this guide](../how-tos/human_in_the_loop/review-tool-calls.ipynb) for a detailed how-to on doing this!

#### Time Travel

When working with agents, we often want closely examine their decision making process: 

(1) Even when they arrive a desired final result, the reasoning that led to that result is often important to examine.

(2) When agents make mistakes, it is often valuable to understand why.

(3) In either of the above cases, it is useful to manually explore alternative decision making paths.

Collectively, we call these debugging concepts `time-travel` and they are composed of `replaying` and `forking`.

##### Replaying

![](./img/human_in_the_loop/replay.png)

Sometimes we want to simply replay past actions of an agent. 
 
Above, we showed the case of executing an agent from the current state (or checkpoint) of the graph.

We by simply passing in `None` for the input with a `thread`.

```
thread = {"configurable": {"thread_id": "1"}}
for event in graph.stream(None, thread, stream_mode="values"):
    print(event)
```

Now, we can modify this to replay past actions from a *specific* checkpoint by passing in the checkpoint ID.

To get a specific checkpoint ID, we can easily get all of the checkpoints in the thread and filter to the one we want.

```python
all_checkpoints = []
for state in app.get_state_history(thread):
    all_checkpoints.append(state)
```

Each checkpoint has a unique ID, which we can use to replay from a specific checkpoint.

Assume from reviewing the checkpoints that we want to replay from one, `xxx`.

We just pass in the checkpoint ID when we run the graph.

```python
config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xxx'}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```
 
Importantly, the graph knows which checkpoints have been previously executed. 

So, it will re-play any previously executed nodes rather than re-executing them.

See [this additional conceptual guide](https://langchain-ai.github.io/langgraph/concepts/persistence/#replay) for related context on replaying.

See see [this guide](../how-tos/human_in_the_loop/time-travel.ipynb) for a detailed how-to on doing time-travel!

##### Forking

![](./img/human_in_the_loop/forking.png)

Sometimes we want to fork past actions of an agent, and explore different paths through the graph.

`Editing`, as discussed above, is *exactly* how we do this for the *current* state of the graph! 

But, what if we want to fork *past* states of the graph?

For example, let's say we want to edit a particular checkpoint, `xxx`.

We pass this `checkpoint_id` when we update the state of the graph.

```python
config = {"configurable": {"thread_id": "1", "checkpoint_id": "xxx"}}
graph.update_state(config, {"state": "updated state"}, )
```

This creates a new forked checkpoint, `xxx-fork`, which we can then run the graph from.

```python
config = {'configurable': {'thread_id': '1', 'checkpoint_id': 'xxx-fork'}}
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```

See [this additional conceptual guide](https://langchain-ai.github.io/langgraph/concepts/persistence/#update-state) for related context on forking.

See [this guide](../how-tos/human_in_the_loop/time-travel.ipynb) for a detailed how-to on doing time-travel!


<a name="docshow-tosauthcustom_authmd"></a>

## How to add custom authentication

!!! tip "Prerequisites"

    This guide assumes familiarity with the following concepts:

      *  [**Authentication & Access Control**](#docsconceptsauthmd)
      *  [**LangGraph Platform**](#langgraph-platform)
    
    For a more guided walkthrough, see [**setting up custom authentication**](#docstutorialsauthgetting_startedmd) tutorial.

???+ note "Python only"
  
    We currently only support custom authentication and authorization in Python deployments with `langgraph-api>=0.0.11`. Support for LangGraph.JS will be added soon.

This guide shows how to add custom authentication to your LangGraph Platform application. This guide applies to both LangGraph Cloud, BYOC, and self-hosted deployments. It does not apply to isolated usage of the LangGraph open source library in your own custom server.

### 1. Implement authentication

Create `auth.py` file, with a basic JWT authentication handler:

```python
from langgraph_sdk import Auth

my_auth = Auth()

@my_auth.authenticate
async def authenticate(authorization: str) -> str:
    token = authorization.split(" ", 1)[-1] # "Bearer <token>"
    try:
        # Verify token with your auth provider
        user_id = await verify_token(token)
        return user_id
    except Exception:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid token"
        )

## Optional: Add authorization rules
@my_auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,
    value: dict,
):
    """Add owner to resource metadata and filter by owner."""
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
```

### 2. Update configuration

In your `langgraph.json`, add the path to your auth file:

```json hl_lines="7-9"
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": ".env",
  "auth": {
    "path": "./auth.py:my_auth"
  }
}
```

### 3. Connect from the client

Once you've set up authentication in your server, requests must include the the required authorization information based on your chosen scheme.
Assuming you are using JWT token authentication, you could access your deployments using any of the following methods:

=== "Python Client"

    ```python
    from langgraph_sdk import get_client

    my_token = "your-token" # In practice, you would generate a signed token with your auth provider
    client = get_client(
        url="http://localhost:2024",
        headers={"Authorization": f"Bearer {my_token}"}
    )
    threads = await client.threads.search()
    ```

=== "Python RemoteGraph"

    ```python
    from langgraph.pregel.remote import RemoteGraph
    
    my_token = "your-token" # In practice, you would generate a signed token with your auth provider
    remote_graph = RemoteGraph(
        "agent",
        url="http://localhost:2024",
        headers={"Authorization": f"Bearer {my_token}"}
    )
    threads = await remote_graph.ainvoke(...)
    ```

=== "JavaScript Client"

    ```javascript
    import { Client } from "@langchain/langgraph-sdk";

    const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
    const client = new Client({
      apiUrl: "http://localhost:2024",
      headers: { Authorization: `Bearer ${my_token}` },
    });
    const threads = await client.threads.search();
    ```

=== "JavaScript RemoteGraph"

    ```javascript
    import { RemoteGraph } from "@langchain/langgraph/remote";

    const my_token = "your-token"; // In practice, you would generate a signed token with your auth provider
    const remoteGraph = new RemoteGraph({
      graphId: "agent",
      url: "http://localhost:2024",
      headers: { Authorization: `Bearer ${my_token}` },
    });
    const threads = await remoteGraph.invoke(...);
    ```

=== "CURL"

    ```bash
    curl -H "Authorization: Bearer ${your-token}" http://localhost:2024/threads
    ```


<a name="docshow-tosauthopenapi_securitymd"></a>

## How to document API authentication in OpenAPI

This guide shows how to customize the OpenAPI security schema for your LangGraph Platform API documentation. A well-documented security schema helps API consumers understand how to authenticate with your API and even enables automatic client generation. See the [Authentication & Access Control conceptual guide](#docsconceptsauthmd) for more details about LangGraph's authentication system.

!!! note "Implementation vs Documentation"
    This guide only covers how to document your security requirements in OpenAPI. To implement the actual authentication logic, see [How to add custom authentication](#docshow-tosauthcustom_authmd).

This guide applies to all LangGraph Platform deployments (Cloud, BYOC, and self-hosted). It does not apply to usage of the LangGraph open source library if you are not using LangGraph Platform.

### Default Schema

The default security scheme varies by deployment type:

=== "LangGraph Cloud"

By default, LangGraph Cloud requires a LangSmith API key in the `x-api-key` header:

```yaml
components:
  securitySchemes:
    apiKeyAuth:
      type: apiKey
      in: header
      name: x-api-key
security:
  - apiKeyAuth: []
```

When using one of the LangGraph SDK's, this can be inferred from environment variables.

=== "Self-hosted"

By default, self-hosted deployments have no security scheme. This means they are to be deployed only on a secured network or with authentication. To add custom authentication, see [How to add custom authentication](#docshow-tosauthcustom_authmd).

### Custom Security Schema

To customize the security schema in your OpenAPI documentation, add an `openapi` field to your `auth` configuration in `langgraph.json`. Remember that this only updates the API documentation - you must also implement the corresponding authentication logic as shown in [How to add custom authentication](#docshow-tosauthcustom_authmd).

Note that LangGraph Platform does not provide authentication endpoints - you'll need to handle user authentication in your client application and pass the resulting credentials to the LangGraph API.

=== "OAuth2 with Bearer Token"

    ```json
    {
      "auth": {
        "path": "./auth.py:my_auth",  // Implement auth logic here
        "openapi": {
          "securitySchemes": {
            "OAuth2": {
              "type": "oauth2",
              "flows": {
                "implicit": {
                  "authorizationUrl": "https://your-auth-server.com/oauth/authorize",
                  "scopes": {
                    "me": "Read information about the current user",
                    "threads": "Access to create and manage threads"
                  }
                }
              }
            }
          },
          "security": [
            {"OAuth2": ["me", "threads"]}
          ]
        }
      }
    }
    ```

=== "API Key"

    ```json
    {
      "auth": {
        "path": "./auth.py:my_auth",  // Implement auth logic here
        "openapi": {
          "securitySchemes": {
            "apiKeyAuth": {
              "type": "apiKey",
              "in": "header",
              "name": "X-API-Key"
            }
          },
          "security": [
            {"apiKeyAuth": []}
          ]
        }
      }
    }
    ```

### Testing

After updating your configuration:

1. Deploy your application
2. Visit `/docs` to see the updated OpenAPI documentation
3. Try out the endpoints using credentials from your authentication server (make sure you've implemented the authentication logic first)


<a name="docshow-tosdeploy-self-hostedmd"></a>

## How to do a Self-hosted deployment of LangGraph

!!! info "Prerequisites"

    - [Application Structure](#docsconceptsapplication_structuremd)
    - [Deployment Options](#docsconceptsdeployment_optionsmd)

This how-to guide will walk you through how to create a docker image from an existing LangGraph application, so you can deploy it on your own infrastructure.

### How it works

With the self-hosted deployment option, you are responsible for managing the infrastructure, including setting up and maintaining necessary databases, Redis instances, and other services.

You will need to do the following:

1. Deploy Redis and Postgres instances on your own infrastructure.
2. Build a docker image with the [LangGraph Server](#docsconceptslanggraph_servermd) using the [LangGraph CLI](#docsconceptslanggraph_climd).
3. Deploy a web server that will run the docker image and pass in the necessary environment variables.

### Helm Chart

If you would like to deploy LangGraph Cloud on Kubernetes, you can use this [Helm chart](https://github.com/langchain-ai/helm/blob/main/charts/langgraph-cloud/README.md).

### Environment Variables

You will eventually need to pass in the following environment variables to the LangGraph Deploy server:

- `REDIS_URI`: Connection details to a Redis instance. Redis will be used as a pub-sub broker to enable streaming real time output from background runs.
- `DATABASE_URI`: Postgres connection details. Postgres will be used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics.
- `LANGSMITH_API_KEY`: (If using [Self-Hosted Lite](#self-hosted-lite)) LangSmith API key. This will be used to authenticate ONCE at server start up.
- `LANGGRAPH_CLOUD_LICENSE_KEY`: (If using [Self-Hosted Enterprise](#self-hosted-enterprise)) LangGraph Platform license key. This will be used to authenticate ONCE at server start up.


### Build the Docker Image

Please read the [Application Structure](#docsconceptsapplication_structuremd) guide to understand how to structure your LangGraph application.

If the application is structured correctly, you can build a docker image with the LangGraph Deploy server.

To build the docker image, you first need to install the CLI:

```shell
pip install -U langgraph-cli
```

You can then use:

```
langgraph build -t my-image
```

This will build a docker image with the LangGraph Deploy server. The `-t my-image` is used to tag the image with a name.

When running this server, you need to pass three environment variables:

### Running the application locally

#### Using Docker

```shell
docker run \
    --env-file .env \
    -p 8123:8000 \
    -e REDIS_URI="foo" \
    -e DATABASE_URI="bar" \
    -e LANGSMITH_API_KEY="baz" \
    my-image
```

If you want to run this quickly without setting up a separate Redis and Postgres instance, you can use this docker compose file.

!!! note

    * You need to replace `my-image` with the name of the image you built in the previous step (from `langgraph build`).
    and you should provide appropriate values for `REDIS_URI`, `DATABASE_URI`, and `LANGSMITH_API_KEY`.
    * If your application requires additional environment variables, you can pass them in a similar way.
    * If using [Self-Hosted Enterprise](#self-hosted-enterprise), you must provide `LANGGRAPH_CLOUD_LICENSE_KEY` as an additional environment variable.


#### Using Docker Compose

```yml
volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: postgres:16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        image: ${IMAGE_NAME}
        ports:
            - "8123:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        env_file:
            - .env
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            LANGSMITH_API_KEY: ${LANGSMITH_API_KEY}
            POSTGRES_URI: postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable
```

You can then run `docker compose up` with this Docker compose file in the same folder.

This will spin up LangGraph Deploy on port `8123` (if you want to change this, you can change this by changing the ports in the `langgraph-api` volume).

You can test that the application is up by checking:

```shell
curl --request GET --url 0.0.0.0:8123/ok
```
Assuming everything is running correctly, you should see a response like:

```shell
{"ok":true}
```



<a name="docshow-tosindexmd"></a>

## How-to Guides

Here youÔÇÖll find answers to ÔÇ£How do I...?ÔÇØ types of questions. These guides are **goal-oriented** and concrete; they're meant to help you complete a specific task. For conceptual explanations see the [Conceptual guide](#docsconceptsindexmd). For end-to-end walk-throughs see [Tutorials](#docstutorialsindexmd). For comprehensive descriptions of every class and function see the [API Reference](#docsreferenceindexmd).

### LangGraph

#### Controllability

LangGraph offers a high level of control over the execution of your graph.

These how-to guides show how to achieve that controllability.

- [How to create branches for parallel execution](branching.ipynb)
- [How to create map-reduce branches for parallel execution](map-reduce.ipynb)
- [How to control graph recursion limit](recursion-limit.ipynb)
- [How to combine control flow and state updates with Command](command.ipynb)

#### Persistence

[LangGraph Persistence](#docsconceptspersistencemd) makes it easy to persist state across graph runs (thread-level persistence) and across threads (cross-thread persistence). These how-to guides show how to add persistence to your graph.

- [How to add thread-level persistence to your graph](persistence.ipynb)
- [How to add thread-level persistence to subgraphs](subgraph-persistence.ipynb)
- [How to add cross-thread persistence to your graph](cross-thread-persistence.ipynb)
- [How to use Postgres checkpointer for persistence](persistence_postgres.ipynb)
- [How to use MongoDB checkpointer for persistence](persistence_mongodb.ipynb)
- [How to create a custom checkpointer using Redis](persistence_redis.ipynb)

#### Memory

LangGraph makes it easy to manage conversation [memory](#docsconceptsmemorymd) in your graph. These how-to guides show how to implement different strategies for that.

- [How to manage conversation history](memory/manage-conversation-history.ipynb)
- [How to delete messages](memory/delete-messages.ipynb)
- [How to add summary conversation memory](memory/add-summary-conversation-history.ipynb)
- [How to add long-term memory (cross-thread)](cross-thread-persistence.ipynb)
- [How to use semantic search for long-term memory](memory/semantic-search.ipynb)

#### Human-in-the-loop

[Human-in-the-loop](#docsconceptshuman_in_the_loopmd) functionality allows
you to involve humans in the decision-making process of your graph. These how-to guides show how to implement human-in-the-loop workflows in your graph.


Key workflows:

- [How to wait for user input](human_in_the_loop/wait-user-input.ipynb): A basic example that shows how to implement a human-in-the-loop workflow in your graph using the `interrupt` function.
- [How to review tool calls](human_in_the_loop/review-tool-calls.ipynb): Incorporate human-in-the-loop for reviewing/editing/accepting tool call requests before they executed using the `interrupt` function.
 

Other methods:

- [How to add static breakpoints](human_in_the_loop/breakpoints.ipynb): Use for debugging purposes. For [**human-in-the-loop**](#docsconceptshuman_in_the_loopmd) workflows, we recommend the [`interrupt` function][langgraph.types.interrupt] instead.
- [How to edit graph state](human_in_the_loop/edit-graph-state.ipynb): Edit graph state using `graph.update_state` method. Use this if implementing a **human-in-the-loop** workflow via **static breakpoints**.
- [How to add dynamic breakpoints with `NodeInterrupt`](human_in_the_loop/dynamic_breakpoints.ipynb): **Not recommended**: Use the [`interrupt` function](#docsconceptshuman_in_the_loopmd) instead.

#### Time Travel

[Time travel](#docsconceptstime-travelmd) allows you to replay past actions in your LangGraph application to explore alternative paths and debug issues. These how-to guides show how to use time travel in your graph.

- [How to view and update past graph state](human_in_the_loop/time-travel.ipynb)

#### Streaming

[Streaming](#docsconceptsstreamingmd) is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of LLMs.

- [How to stream full state of your graph](stream-values.ipynb)
- [How to stream state updates of your graph](stream-updates.ipynb)
- [How to stream LLM tokens](streaming-tokens.ipynb)
- [How to stream LLM tokens without LangChain models](streaming-tokens-without-langchain.ipynb)
- [How to stream custom data](streaming-content.ipynb)
- [How to configure multiple streaming modes at the same time](stream-multiple.ipynb)
- [How to stream events from within a tool](streaming-events-from-within-tools.ipynb)
- [How to stream events from within a tool without LangChain models](streaming-events-from-within-tools-without-langchain.ipynb)
- [How to stream events from the final node](streaming-from-final-node.ipynb)
- [How to stream from subgraphs](streaming-subgraphs.ipynb)
- [How to disable streaming for models that don't support it](disable-streaming.ipynb)

#### Tool calling

[Tool calling](https://python.langchain.com/docs/concepts/tool_calling/) is a type of chat model API that accepts tool schemas, along with messages, as input and returns invocations of those tools as part of the output message.

These how-to guides show common patterns for tool calling with LangGraph:

- [How to call tools using ToolNode](tool-calling.ipynb)
- [How to handle tool calling errors](tool-calling-errors.ipynb)
- [How to pass runtime values to tools](pass-run-time-values-to-tools.ipynb)
- [How to pass config to tools](pass-config-to-tools.ipynb)
- [How to update graph state from tools](update-state-from-tools.ipynb)
- [How to handle large numbers of tools](many-tools.ipynb)

#### Subgraphs

[Subgraphs](#subgraphs) allow you to reuse an existing graph from another graph. These how-to guides show how to use subgraphs:

- [How to add and use subgraphs](subgraph.ipynb)
- [How to view and update state in subgraphs](subgraphs-manage-state.ipynb)
- [How to transform inputs and outputs of a subgraph](subgraph-transform-state.ipynb)

#### Multi-agent

[Multi-agent systems](#docsconceptsmulti_agentmd) are useful to break down complex LLM applications into multiple agents, each responsible for a different part of the application. These how-to guides show how to implement multi-agent systems in LangGraph:

- [How to implement handoffs between agents](agent-handoffs.ipynb)
- [How to build a multi-agent network](multi-agent-network.ipynb)
- [How to add multi-turn conversation in a multi-agent application](multi-agent-multi-turn-convo.ipynb)

See the [multi-agent tutorials](#multi-agent-systems) for implementations of other multi-agent architectures.

#### State Management

- [How to use Pydantic model as state](state-model.ipynb)
- [How to define input/output schema for your graph](input_output_schema.ipynb)
- [How to pass private state between nodes inside the graph](pass_private_state.ipynb)

#### Other

- [How to run graph asynchronously](async.ipynb)
- [How to visualize your graph](visualization.ipynb)
- [How to add runtime configuration to your graph](configuration.ipynb)
- [How to add node retries](node-retries.ipynb)
- [How to force function calling agent to structure output](react-agent-structured-output.ipynb)
- [How to pass custom LangSmith run ID for graph runs](run-id-langsmith.ipynb)
- [How to return state before hitting recursion limit](return-when-recursion-limit-hits.ipynb)
- [How to integrate LangGraph with AutoGen, CrewAI, and other frameworks](autogen-integration.ipynb)

#### Prebuilt ReAct Agent

The LangGraph [prebuilt ReAct agent](#langgraph.prebuilt.chat_agent_executor.create_react_agent) is pre-built implementation of a [tool calling agent](#tool-calling-agent).

One of the big benefits of LangGraph is that you can easily create your own agent architectures. So while it's fine to start here to build an agent quickly, we would strongly recommend learning how to build your own agent so that you can take full advantage of LangGraph.

These guides show how to use the prebuilt ReAct agent:

- [How to create a ReAct agent](create-react-agent.ipynb)
- [How to add memory to a ReAct agent](create-react-agent-memory.ipynb)
- [How to add a custom system prompt to a ReAct agent](create-react-agent-system-prompt.ipynb)
- [How to add human-in-the-loop processes to a ReAct agent](create-react-agent-hitl.ipynb)
- [How to create prebuilt ReAct agent from scratch](react-agent-from-scratch.ipynb)
- [How to add semantic search for long-term memory to a ReAct agent](#using-in-create-react-agent)

### LangGraph Platform

This section includes how-to guides for LangGraph Platform.

LangGraph Platform is a commercial solution for deploying agentic applications in production, built on the open-source LangGraph framework.

The LangGraph Platform offers a few different deployment options described in the [deployment options guide](#docsconceptsdeployment_optionsmd).

!!! tip

    * LangGraph is an MIT-licensed open-source library, which we are committed to maintaining and growing for the community.
    * You can always deploy LangGraph applications on your own infrastructure using the open-source LangGraph project without using LangGraph Platform.

#### Application Structure

Learn how to set up your app for deployment to LangGraph Platform:

- [How to set up app for deployment (requirements.txt)](#docsclouddeploymentsetupmd)
- [How to set up app for deployment (pyproject.toml)](#docsclouddeploymentsetup_pyprojectmd)
- [How to set up app for deployment (JavaScript)](#docsclouddeploymentsetup_javascriptmd)
- [How to add semantic search](#docsclouddeploymentsemantic_searchmd)
- [How to customize Dockerfile](#docsclouddeploymentcustom_dockermd)
- [How to test locally](#docsclouddeploymenttest_locallymd)
- [How to rebuild graph at runtime](#docsclouddeploymentgraph_rebuildmd)
- [How to use LangGraph Platform to deploy CrewAI, AutoGen, and other frameworks](autogen-langgraph-platform.ipynb)

#### Deployment

LangGraph applications can be deployed using LangGraph Cloud, which provides a range of services to help you deploy, manage, and scale your applications.

- [How to deploy to LangGraph cloud](#docsclouddeploymentcloudmd)
- [How to deploy to a self-hosted environment](#docshow-tosdeploy-self-hostedmd)
- [How to interact with the deployment using RemoteGraph](#docshow-tosuse-remote-graphmd)

#### Authentication & Access Control

- [How to add custom authentication](#docshow-tosauthcustom_authmd)
- [How to update the security schema of your OpenAPI spec](#docshow-tosauthopenapi_securitymd)

#### Assistants

[Assistants](#docsconceptsassistantsmd) is a configured instance of a template.

- [How to configure agents](#docscloudhow-tosconfiguration_cloudmd)
- [How to version assistants](#docscloudhow-tosassistant_versioningmd)

#### Threads

- [How to copy threads](#docscloudhow-toscopy_threadsmd)
- [How to check status of your threads](#docscloudhow-toscheck_thread_statusmd)

#### Runs

LangGraph Platform supports multiple types of runs besides streaming runs.

- [How to run an agent in the background](#docscloudhow-tosbackground_runmd)
- [How to run multiple agents in the same thread](#docscloudhow-tossame-threadmd)
- [How to create cron jobs](#docscloudhow-toscron_jobsmd)
- [How to create stateless runs](#docscloudhow-tosstateless_runsmd)

#### Streaming

Streaming the results of your LLM application is vital for ensuring a good user experience, especially when your graph may call multiple models and take a long time to fully complete a run. Read about how to stream values from your graph in these how to guides:

- [How to stream values](#docscloudhow-tosstream_valuesmd)
- [How to stream updates](#docscloudhow-tosstream_updatesmd)
- [How to stream messages](#docscloudhow-tosstream_messagesmd)
- [How to stream events](#docscloudhow-tosstream_eventsmd)
- [How to stream in debug mode](#docscloudhow-tosstream_debugmd)
- [How to stream multiple modes](#docscloudhow-tosstream_multiplemd)

#### Human-in-the-loop

When designing complex graphs, relying entirely on the LLM for decision-making can be risky, particularly when it involves tools that interact with files, APIs, or databases. These interactions may lead to unintended data access or modifications, depending on the use case. To mitigate these risks, LangGraph allows you to integrate human-in-the-loop behavior, ensuring your LLM applications operate as intended without undesirable outcomes.

- [How to add a breakpoint](#docscloudhow-toshuman_in_the_loop_breakpointmd)
- [How to wait for user input](#docscloudhow-toshuman_in_the_loop_user_inputmd)
- [How to edit graph state](#docscloudhow-toshuman_in_the_loop_edit_statemd)
- [How to replay and branch from prior states](#docscloudhow-toshuman_in_the_loop_time_travelmd)
- [How to review tool calls](#docscloudhow-toshuman_in_the_loop_review_tool_callsmd)

#### Double-texting

Graph execution can take a while, and sometimes users may change their mind about the input they wanted to send before their original input has finished running. For example, a user might notice a typo in their original request and will edit the prompt and resend it. Deciding what to do in these cases is important for ensuring a smooth user experience and preventing your graphs from behaving in unexpected ways.

- [How to use the interrupt option](#docscloudhow-tosinterrupt_concurrentmd)
- [How to use the rollback option](#docscloudhow-tosrollback_concurrentmd)
- [How to use the reject option](#docscloudhow-tosreject_concurrentmd)
- [How to use the enqueue option](#docscloudhow-tosenqueue_concurrentmd)

#### Webhooks

- [How to integrate webhooks](#docscloudhow-toswebhooksmd)

#### Cron Jobs

- [How to create cron jobs](#docscloudhow-toscron_jobsmd)

#### LangGraph Studio

LangGraph Studio is a built-in UI for visualizing, testing, and debugging your agents.

- [How to connect to a LangGraph Cloud deployment](#docscloudhow-tostest_deploymentmd)
- [How to connect to a local dev server](#docshow-toslocal-studiomd)
- [How to connect to a local deployment (Docker)](#docscloudhow-tostest_local_deploymentmd)
- [How to test your graph in LangGraph Studio (MacOS only)](#docscloudhow-tosinvoke_studiomd)
- [How to interact with threads in LangGraph Studio](#docscloudhow-tosthreads_studiomd)
- [How to add nodes as dataset examples in LangGraph Studio](#docscloudhow-tosdatasets_studiomd)

### Troubleshooting

These are the guides for resolving common errors you may find while building with LangGraph. Errors referenced below will have an `lc_error_code` property corresponding to one of the below codes when they are thrown in code.

- [GRAPH_RECURSION_LIMIT](#docstroubleshootingerrorsgraph_recursion_limitmd)
- [INVALID_CONCURRENT_GRAPH_UPDATE](#docstroubleshootingerrorsinvalid_concurrent_graph_updatemd)
- [INVALID_GRAPH_NODE_RETURN_VALUE](#docstroubleshootingerrorsinvalid_graph_node_return_valuemd)
- [MULTIPLE_SUBGRAPHS](#docstroubleshootingerrorsmultiple_subgraphsmd)
- [INVALID_CHAT_HISTORY](#docstroubleshootingerrorsinvalid_chat_historymd)


<a name="docshow-toslocal-studiomd"></a>

## How to connect a local agent to LangGraph Studio

This guide shows you how to connect your local agent to [LangGraph Studio](#docsconceptslanggraph_studiomd) for visualization, interaction, and debugging.

### Connection Options

There are two ways to connect your local agent to LangGraph Studio:

- [Development Server](#development-server-with-web-ui): Python package, all platforms, no Docker
- [LangGraph Desktop](#desktop-app): Application, Mac only, requires Docker

In this guide we will cover how to use the development server as that is generally an easier and better experience.

### Setup your application

First, you will need to setup your application in the proper format.
This means defining a `langgraph.json` file which contains paths to your agent(s).
See [this guide](#docsconceptsapplication_structuremd) for information on how to do so.

### Install langgraph-cli

You will need to install [`langgraph-cli`](#langgraph-cli) (version `0.1.55` or higher).
You will need to make sure to install the `inmem` extras.

???+ note "Minimum version"

  The minimum version to use the `inmem` extra with `langgraph-cli` is `0.1.55`.
  Python 3.11 or higher is required.
  

```shell
pip install -U "langgraph-cli[inmem]"
```

### Run the development server

1. Navigate to your project directory (where `langgraph.json` is located)

2. Start the server:
   ```bash
   langgraph dev
   ```

This will look for the `langgraph.json` file in your current directory. 
In there, it will find the paths to the graph(s), and start those up.
It will then automatically connect to the cloud-hosted studio.

### Use the studio

After connecting to the studio, a browser window should automatically pop up.
This will use the cloud hosted studio UI to connect to your local development server.
Your graph is still running locally, the UI is connecting to visualizing the agent and threads that are defined locally.

The graph will always use the most up-to-date code, so you will be able to change the underlying code and have it automatically reflected in the studio.
This is useful for debugging workflows.
You can run your graph in the UI until it messes up, go in and change your code, and then rerun from the node that failed.

## (Optional) Attach a debugger

For step-by-step debugging with breakpoints and variable inspection:

```bash
## Install debugpy package
pip install debugpy

## Start server with debugging enabled
langgraph dev --debug-port 5678
```

Then attach your preferred debugger:

=== "VS Code"
    Add this configuration to `launch.json`:
    ```json
    {
      "name": "Attach to LangGraph",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "0.0.0.0",
        "port": 5678
      }
    }
    ```
    Specify the port number you chose in the previous step.

=== "PyCharm"
    1. Go to Run ÔåÆ Edit Configurations
    2. Click + and select "Python Debug Server"
    3. Set IDE host name: `localhost`
    4. Set port: `5678` (or the port number you chose in the previous step)
    5. Click "OK" and start debugging

<a name="docshow-tosuse-remote-graphmd"></a>

## How to interact with the deployment using RemoteGraph

!!! info "Prerequisites"
    - [LangGraph Platform](#docsconceptslanggraph_platformmd)
    - [LangGraph Server](#docsconceptslanggraph_servermd)

`RemoteGraph` is an interface that allows you to interact with your LangGraph Platform deployment as if it were a regular, locally-defined LangGraph graph (e.g. a `CompiledGraph`). This guide shows you how you can initialize a `RemoteGraph` and interact with it.

### Initializing the graph

When initializing a `RemoteGraph`, you must always specify:

- `name`: the name of the graph you want to interact with. This is the same graph name you use in `langgraph.json` configuration file for your deployment. 
- `api_key`: a valid LangSmith API key. Can be set as an environment variable (`LANGSMITH_API_KEY`) or passed directly via the `api_key` argument. The API key could also be provided via the `client` / `sync_client` arguments, if `LangGraphClient` / `SyncLangGraphClient` were initialized with `api_key` argument.

Additionally, you have to provide one of the following:

- `url`: URL of the deployment you want to interact with. If you pass `url` argument, both sync and async clients will be created using the provided URL, headers (if provided) and default configuration values (e.g. timeout, etc).
- `client`: a `LangGraphClient` instance for interacting with the deployment asynchronously (e.g. using `.astream()`, `.ainvoke()`, `.aget_state()`, `.aupdate_state()`, etc.)
- `sync_client`: a `SyncLangGraphClient` instance for interacting with the deployment synchronously (e.g. using `.stream()`, `.invoke()`, `.get_state()`, `.update_state()`, etc.)

!!! Note

    If you pass both `client` or `sync_client` as well as `url` argument, they will take precedence over the `url` argument. If none of the `client` / `sync_client` / `url` arguments are provided, `RemoteGraph` will raise a `ValueError` at runtime.


#### Using URL

=== "Python"

    ```python
    from langgraph.pregel.remote import RemoteGraph

    url = <DEPLOYMENT_URL>
    graph_name = "agent"
    remote_graph = RemoteGraph(graph_name, url=url)
    ```

=== "JavaScript"

    ```ts
    import { RemoteGraph } from "@langchain/langgraph/remote";

    const url = `<DEPLOYMENT_URL>`;
    const graphName = "agent";
    const remoteGraph = new RemoteGraph({ graphId: graphName, url });
    ```

#### Using clients

=== "Python"

    ```python
    from langgraph_sdk import get_client, get_sync_client
    from langgraph.pregel.remote import RemoteGraph

    url = <DEPLOYMENT_URL>
    graph_name = "agent"
    client = get_client(url=url)
    sync_client = get_sync_client(url=url)
    remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)
    ```

=== "JavaScript"

    ```ts
    import { Client } from "@langchain/langgraph-sdk";
    import { RemoteGraph } from "@langchain/langgraph/remote";

    const client = new Client({ apiUrl: `<DEPLOYMENT_URL>` });
    const graphName = "agent";
    const remoteGraph = new RemoteGraph({ graphId: graphName, client });
    ```

### Invoking the graph

Since `RemoteGraph` is a `Runnable` that implements the same methods as `CompiledGraph`, you can interact with it the same way you normally would with a compiled graph, i.e. by calling `.invoke()`, `.stream()`, `.get_state()`, `.update_state()`, etc (as well as their async counterparts).

#### Asynchronously

!!! Note

    To use the graph asynchronously, you must provide either the `url` or `client` when initializing the `RemoteGraph`.

=== "Python"

    ```python
    # invoke the graph
    result = await remote_graph.ainvoke({
        "messages": [{"role": "user", "content": "what's the weather in sf"}]
    })

    # stream outputs from the graph
    async for chunk in remote_graph.astream({
        "messages": [{"role": "user", "content": "what's the weather in la"}]
    }):
        print(chunk)
    ```

=== "JavaScript"

    ```ts
    // invoke the graph
    const result = await remoteGraph.invoke({
        messages: [{role: "user", content: "what's the weather in sf"}]
    })

    // stream outputs from the graph
    for await (const chunk of await remoteGraph.stream({
        messages: [{role: "user", content: "what's the weather in la"}]
    })):
        console.log(chunk)
    ```

#### Synchronously

!!! Note

    To use the graph synchronously, you must provide either the `url` or `sync_client` when initializing the `RemoteGraph`.

=== "Python"

    ```python
    # invoke the graph
    result = remote_graph.invoke({
        "messages": [{"role": "user", "content": "what's the weather in sf"}]
    })

    # stream outputs from the graph
    for chunk in remote_graph.stream({
        "messages": [{"role": "user", "content": "what's the weather in la"}]
    }):
        print(chunk)
    ```

### Thread-level persistence

By default, the graph runs (i.e. `.invoke()` or `.stream()` invocations) are stateless - the checkpoints and the final state of the graph are not persisted. If you would like to persist the outputs of the graph run (for example, to enable human-in-the-loop features), you can create a thread and provide the thread ID via the `config` argument, same as you would with a regular compiled graph:

=== "Python"

    ```python
    from langgraph_sdk import get_sync_client
    url = <DEPLOYMENT_URL>
    graph_name = "agent"
    sync_client = get_sync_client(url=url)
    remote_graph = RemoteGraph(graph_name, url=url)

    # create a thread (or use an existing thread instead)
    thread = sync_client.threads.create()

    # invoke the graph with the thread config
    config = {"configurable": {"thread_id": thread["thread_id"]}}
    result = remote_graph.invoke({
        "messages": [{"role": "user", "content": "what's the weather in sf"}]
    }, config=config)

    # verify that the state was persisted to the thread
    thread_state = remote_graph.get_state(config)
    print(thread_state)
    ```

=== "JavaScript"

    ```ts
    import { Client } from "@langchain/langgraph-sdk";
    import { RemoteGraph } from "@langchain/langgraph/remote";

    const url = `<DEPLOYMENT_URL>`;
    const graphName = "agent";
    const client = new Client({ apiUrl: url });
    const remoteGraph = new RemoteGraph({ graphId: graphName, url });

    // create a thread (or use an existing thread instead)
    const thread = await client.threads.create();

    // invoke the graph with the thread config
    const config = { configurable: { thread_id: thread.thread_id }};
    const result = await remoteGraph.invoke({
      messages: [{ role: "user", content: "what's the weather in sf" }],
    }, config);

    // verify that the state was persisted to the thread
    const threadState = await remoteGraph.getState(config);
    console.log(threadState);
    ```

### Using as a subgraph

!!! Note

    If you need to use a `checkpointer` with a graph that has a `RemoteGraph` subgraph node, make sure to use UUIDs as thread IDs.


Since the `RemoteGraph` behaves the same way as a regular `CompiledGraph`, it can be also used as a subgraph in another graph. For example:

=== "Python"

    ```python
    from langgraph_sdk import get_sync_client
    from langgraph.graph import StateGraph, MessagesState, START
    from typing import TypedDict

    url = <DEPLOYMENT_URL>
    graph_name = "agent"
    remote_graph = RemoteGraph(graph_name, url=url)

    # define parent graph
    builder = StateGraph(MessagesState)
    # add remote graph directly as a node
    builder.add_node("child", remote_graph)
    builder.add_edge(START, "child")
    graph = builder.compile()

    # invoke the parent graph
    result = graph.invoke({
        "messages": [{"role": "user", "content": "what's the weather in sf"}]
    })
    print(result)

    # stream outputs from both the parent graph and subgraph
    for chunk in graph.stream({
        "messages": [{"role": "user", "content": "what's the weather in sf"}]
    }, subgraphs=True):
        print(chunk)
    ```

=== "JavaScript"

    ```ts
    import { MessagesAnnotation, StateGraph, START } from "@langchain/langgraph";
    import { RemoteGraph } from "@langchain/langgraph/remote";

    const url = `<DEPLOYMENT_URL>`;
    const graphName = "agent";
    const remoteGraph = new RemoteGraph({ graphId: graphName, url });

    // define parent graph and add remote graph directly as a node
    const graph = new StateGraph(MessagesAnnotation)
      .addNode("child", remoteGraph)
      .addEdge(START, "child")
      .compile()

    // invoke the parent graph
    const result = await graph.invoke({
      messages: [{ role: "user", content: "what's the weather in sf" }]
    });
    console.log(result);

    // stream outputs from both the parent graph and subgraph
    for await (const chunk of await graph.stream({
      messages: [{ role: "user", content: "what's the weather in la" }]
    }, { subgraphs: true })) {
      console.log(chunk);
    }
    ```

<a name="docsindexmd"></a>

{!README.md!}


<a name="docsreferencechannelsmd"></a>

## Channels

::: langgraph.channels.base
    options:
      members:
        - BaseChannel

::: langgraph.channels
    options:
      members:
        - Topic
        - LastValue
        - EphemeralValue
        - BinaryOperatorAggregate
        - AnyValue


<a name="docsreferencecheckpointsmd"></a>

## Checkpointers

::: langgraph.checkpoint.base
    options:
      members:
        - CheckpointMetadata
        - Checkpoint
        - BaseCheckpointSaver
        - create_checkpoint

::: langgraph.checkpoint.serde.base
    options:
      members:
        - SerializerProtocol

::: langgraph.checkpoint.serde.jsonplus
    options:
      members:
        - JsonPlusSerializer

::: langgraph.checkpoint.memory

::: langgraph.checkpoint.sqlite

::: langgraph.checkpoint.sqlite.aio

::: langgraph.checkpoint.postgres

::: langgraph.checkpoint.postgres.aio

<a name="docsreferenceconstantsmd"></a>

::: langgraph.constants
    options:
      members:
        - TAG_HIDDEN
        - START
        - END

<a name="docsreferenceerrorsmd"></a>

## Errors

::: langgraph.errors

<a name="docsreferencegraphsmd"></a>

## Graph Definitions

::: langgraph.graph.graph
    options:
      members:
        - Graph
        - CompiledGraph

::: langgraph.graph.state
    options:
      members:
        - StateGraph
        - CompiledStateGraph

::: langgraph.graph.message
    options:
      members:
        - add_messages

<a name="docsreferenceindexmd"></a>

<style>
.md-sidebar {
  display: block !important;
}
</style>


## Reference

Welcome to the LangGraph API reference! This reference provides detailed information about the LangGraph API, including classes, methods, and other components.

If you are new to LangGraph, we recommend starting with the [Quick Start](../tutorials/introduction.ipynb) in the Tutorials section.

<a name="docsreferenceprebuiltmd"></a>

## Prebuilt

::: langgraph.prebuilt.chat_agent_executor
    options:
      members:
        - create_react_agent

::: langgraph.prebuilt.tool_node
    options:
      members:
        - ToolNode
        - InjectedState
        - InjectedStore
        - tools_condition

::: langgraph.prebuilt.tool_validator
    options:
      members:
        - ValidationNode


<a name="docsreferenceremote_graphmd"></a>

## RemoteGraph

::: langgraph.pregel.remote
    options:
      members:
        - RemoteGraph


<a name="docsreferencestoremd"></a>

## Storage

::: langgraph.store.base
    

::: langgraph.store.postgres

<a name="docsreferencetypesmd"></a>

## Types

::: langgraph.types
    options:
      members:
        - All
        - StreamMode
        - StreamWriter
        - RetryPolicy
        - CachePolicy
        - Interrupt
        - PregelTask
        - PregelExecutableTask
        - StateSnapshot
        - Send
        - Command
        - interrupt


<a name="docstroubleshootingerrorsgraph_recursion_limitmd"></a>

## GRAPH_RECURSION_LIMIT

Your LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph) reached the maximum number of steps before hitting a stop condition.
This is often due to an infinite loop caused by code like the example below:

```python
class State(TypedDict):
    some_key: str

builder = StateGraph(State)
builder.add_node("a", ...)
builder.add_node("b", ...)
builder.add_edge("a", "b")
builder.add_edge("b", "a")
...

graph = builder.compile()
```

However, complex graphs may hit the default limit naturally.

### Troubleshooting

- If you are not expecting your graph to go through many iterations, you likely have a cycle. Check your logic for infinite loops.
- If you have a complex graph, you can pass in a higher `recursion_limit` value into your `config` object when invoking your graph like this:

```python
graph.invoke({...}, {"recursion_limit": 100})
```

<a name="docstroubleshootingerrorsinvalid_chat_historymd"></a>

## INVALID_CHAT_HISTORY

This error is raised in the prebuilt [create_react_agent][langgraph.prebuilt.chat_agent_executor.create_react_agent] when the `call_model` graph node receives a malformed list of messages. Specifically, it is malformed when there are `AIMessages` with `tool_calls` (LLM requesting to call a tool) that do not have a corresponding `ToolMessage` (result of a tool invocation to return to the LLM).

There could be a few reasons you're seeing this error:

1. You manually passed a malformed list of messages when invoking the graph, e.g. `graph.invoke({'messages': [AIMessage(..., tool_calls=[...])]})`
2. The graph was interrupted before receiving updates from the `tools` node (i.e. a list of ToolMessages)
and you invoked it with a an input that is not None or a ToolMessage,
e.g. `graph.invoke({'messages': [HumanMessage(...)]}, config)`.
    This interrupt could have been triggered in one of the following ways:
     - You manually set `interrupt_before = ['tools']` in `create_react_agent`
     - One of the tools raised an error that wasn't handled by the [ToolNode][langgraph.prebuilt.tool_node.ToolNode] (`"tools"`)

### Troubleshooting

To resolve this, you can do one of the following:

1. Don't invoke the graph with a malformed list of messages
2. In case of an interrupt (manual or due to an error) you can:

    - provide ToolMessages that match existing tool calls and call `graph.invoke({'messages': [ToolMessage(...)]})`.
    **NOTE**: this will append the messages to the history and run the graph from the START node.
    - manually update the state and resume the graph from the interrupt:

        1. get the list of most recent messages from the graph state with `graph.get_state(config)`
        2. modify the list of messages to either remove unanswered tool calls from AIMessages
or add ToolMessages with tool_call_ids that match unanswered tool calls
        3. call `graph.update_state(config, {'messages': ...})` with the modified list of messages
        4. resume the graph, e.g. call `graph.invoke(None, config)`


<a name="docstroubleshootingerrorsinvalid_concurrent_graph_updatemd"></a>

## INVALID_CONCURRENT_GRAPH_UPDATE

A LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph) received concurrent updates to its state from multiple nodes to a state property that doesn't
support it.

One way this can occur is if you are using a [fanout](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
or other parallel execution in your graph and you have defined a graph like this:

```python
class State(TypedDict):
    some_key: str

def node(state: State):
    return {"some_key": "some_string_value"}

def other_node(state: State):
    return {"some_key": "some_string_value"}


builder = StateGraph(State)
builder.add_node(node)
builder.add_node(other_node)
builder.add_edge(START, "node")
builder.add_edge(START, "other_node")
graph = builder.compile()
```

If a node in the above graph returns `{ "some_key": "some_string_value" }`, this will overwrite the state value for `"some_key"` with `"some_string_value"`.
However, if multiple nodes in e.g. a fanout within a single step return values for `"some_key"`, the graph will throw this error because
there is uncertainty around how to update the internal state.

To get around this, you can define a reducer that combines multiple values:

```python
import operator
from typing import Annotated

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    some_key: Annotated[list, operator.add]
```

This will allow you to define logic that handles the same key returned from multiple nodes executed in parallel.

### Troubleshooting

The following may help resolve this error:

- If your graph executes nodes in parallel, make sure you have defined relevant state keys with a reducer.

<a name="docstroubleshootingerrorsinvalid_graph_node_return_valuemd"></a>

## INVALID_GRAPH_NODE_RETURN_VALUE

A LangGraph [`StateGraph`](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph)
received a non-dict return type from a node. Here's an example:

```python
class State(TypedDict):
    some_key: str

def bad_node(state: State):
    # Should return an dict with a value for "some_key", not a list
    return ["whoops"]

builder = StateGraph(State)
builder.add_node(bad_node)
...

graph = builder.compile()
```

Invoking the above graph will result in an error like this:

```python
graph.invoke({ "some_key": "someval" });
```

```
InvalidUpdateError: Expected dict, got ['whoops']
For troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/INVALID_GRAPH_NODE_RETURN_VALUE
```

Nodes in your graph must return an dict containing one or more keys defined in your state.

### Troubleshooting

The following may help resolve this error:

- If you have complex logic in your node, make sure all code paths return an appropriate dict for your defined state.

<a name="docstroubleshootingerrorsmultiple_subgraphsmd"></a>

## MULTIPLE_SUBGRAPHS

You are calling the same subgraph multiple times within a single LangGraph node with checkpointing enabled for each subgraph.

This is currently not allowed due to internal restrictions on how checkpoint namespacing for subgraphs works.

### Troubleshooting

The following may help resolve this error:

- If you don't need to interrupt/resume from a subgraph, pass `checkpointer=False` when compiling it like this: `.compile(checkpointer=False)`
- Don't imperatively call graphs multiple times in the same node, and instead use the [`Send`](https://langchain-ai.github.io/langgraph/concepts/low_level/#send) API.

<a name="docstroubleshootingerrorsindexmd"></a>

## Error reference

This page contains guides around resolving common errors you may find while building with LangChain.
Errors referenced below will have an `lc_error_code` property corresponding to one of the below codes when they are thrown in code.

- [GRAPH_RECURSION_LIMIT](#docstroubleshootingerrorsgraph_recursion_limitmd)
- [INVALID_CONCURRENT_GRAPH_UPDATE](#docstroubleshootingerrorsinvalid_concurrent_graph_updatemd)
- [INVALID_GRAPH_NODE_RETURN_VALUE](#docstroubleshootingerrorsinvalid_graph_node_return_valuemd)
- [MULTIPLE_SUBGRAPHS](#docstroubleshootingerrorsmultiple_subgraphsmd)
- [INVALID_CHAT_HISTORY](#docstroubleshootingerrorsinvalid_chat_historymd)


<a name="docstutorialsauthadd_auth_servermd"></a>

## Connecting an Authentication Provider (Part 3/3)

!!! note "This is part 3 of our authentication series:"

    1. [Basic Authentication](#docstutorialsauthgetting_startedmd) - Control who can access your bot
    2. [Resource Authorization](#docstutorialsauthresource_authmd) - Let users have private conversations
    3. Production Auth (you are here) - Add real user accounts and validate using OAuth2

In the [Making Conversations Private](#docstutorialsauthresource_authmd) tutorial, we added [resource authorization](#resource-authorization) to give users private conversations. However, we were still using hard-coded tokens for authentication, which is not secure. Now we'll replace those tokens with real user accounts using [OAuth2](#oauth2-authentication).

We'll keep the same [`Auth`](#langgraph_sdk.auth.Auth) object and [resource-level access control](#resource-level-access-control), but upgrade our authentication to use Supabase as our identity provider. While we use Supabase in this tutorial, the concepts apply to any OAuth2 provider. You'll learn how to:

1. Replace test tokens with real [JWT tokens](#jwt-tokens)
2. Integrate with OAuth2 providers for secure user authentication
3. Handle user sessions and metadata while maintaining our existing authorization logic

### Requirements

You will need to set up a Supabase project to use its authentication server for this tutorial. You can do so [here](https://supabase.com/dashboard).

### Background

OAuth2 involves three main roles:

1. **Authorization server**: The identity provider (e.g., Supabase, Auth0, Google) that handles user authentication and issues tokens
2. **Application backend**: Your LangGraph application. This validates tokens and serves protected resources (conversation data)
3. **Client application**: The web or mobile app where users interact with your service

A standard OAuth2 flow works something like this:


```mermaid
sequenceDiagram
    participant User
    participant Client
    participant AuthServer
    participant LangGraph Backend

    User->>Client: Initiate login
    User->>AuthServer: Enter credentials
    AuthServer->>Client: Send tokens
    Client->>LangGraph Backend: Request with token
    LangGraph Backend->>AuthServer: Validate token
    AuthServer->>LangGraph Backend: Token valid
    LangGraph Backend->>Client: Serve request (e.g., run agent or graph)
```

In the following example, we'll use Supabase as our auth server. The LangGraph application will provide the backend for your app, and we will write test code for the client app.
Let's get started!

### Setting Up Authentication Provider {#setup-auth-provider}

First, let's install the required dependencies. Start in your `custom-auth` directory and ensure you have the `langgraph-cli` installed:

```bash
cd custom-auth
pip install -U "langgraph-cli[inmem]"
```

Next, we'll need to fech the URL of our auth server and the private key for authentication.
Since we're using Supabase for this, we can do this in the Supabase dashboard:

1. In the left sidebar, click on t´©ÅÔÜÖ Project Settings" and then click "API"
2. Copy your project URL and add it to your `.env` file

```shell
echo "SUPABASE_URL=your-project-url" >> .env
```
3. Next, copy your service role secret key and add it to your `.env` file
```shell
echo "SUPABASE_SERVICE_KEY=your-service-role-key" >> .env
```
4. Finally, copy your "anon public" key and note it down. This will be used later when we set up our client code.

```bash
SUPABASE_URL=your-project-url
SUPABASE_SERVICE_KEY=your-service-role-key
```

### Implementing Token Validation

In the previous tutorials, we used the [`Auth`](#langgraph_sdk.auth.Auth) object to:

1. Validate hard-coded tokens in the [authentication tutorial](#docstutorialsauthgetting_startedmd)
2. Add resource ownership in the [authorization tutorial](#docstutorialsauthresource_authmd)

Now we'll upgrade our authentication to validate real JWT tokens from Supabase. The key changes will all be in the [`@auth.authenticate`](#langgraph_sdk.auth.Auth.authenticate) decorated function:

1. Instead of checking against a hard-coded list of tokens, we'll make an HTTP request to Supabase to validate the token
2. We'll extract real user information (ID, email) from the validated token

And we'll keep our existing resource authorization logic unchanged

Let's update `src/security/auth.py` to implement this:

```python hl_lines="8-9 20-30" title="src/security/auth.py"
import os
import httpx
from langgraph_sdk import Auth

auth = Auth()

## This is loaded from the `.env` file you created above
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


@auth.authenticate
async def get_current_user(authorization: str | None):
    """Validate JWT tokens and extract user information."""
    assert authorization
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"

    try:
        # Verify token with auth provider
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": authorization,
                    "apiKey": SUPABASE_SERVICE_KEY,
                },
            )
            assert response.status_code == 200
            user = response.json()
            return {
                "identity": user["id"],  # Unique user identifier
                "email": user["email"],
                "is_authenticated": True,
            }
    except Exception as e:
        raise Auth.exceptions.HTTPException(status_code=401, detail=str(e))

## ... the rest is the same as before

## Keep our resource authorization from the previous tutorial
@auth.on
async def add_owner(ctx, value):
    """Make resources private to their creator using resource metadata."""
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
```

The most important change is that we're now validating tokens with a real authentication server. Our authentication handler has the private key for our Supabase project, which we can use to validate the user's token and extract their information.

Let's test this with a real user account!

### Testing Authentication Flow

Let's test out our new authentication flow. You can run the following code in a file or notebook. You will need to provide:

- A valid email address
- A Supabase project URL (from [above](#setup-auth-provider))
- A Supabase anon **public key** (also from [above](#setup-auth-provider))

```python
import os
import httpx
from getpass import getpass
from langgraph_sdk import get_client


## Get email from command line
email = getpass("Enter your email: ")
base_email = email.split("@")
password = "secure-password"  # CHANGEME
email1 = f"{base_email[0]}+1@{base_email[1]}"
email2 = f"{base_email[0]}+2@{base_email[1]}"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
if not SUPABASE_URL:
    SUPABASE_URL = getpass("Enter your Supabase project URL: ")

## This is your PUBLIC anon key (which is safe to use client-side)
## Do NOT mistake this for the secret service role key
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
if not SUPABASE_ANON_KEY:
    SUPABASE_ANON_KEY = getpass("Enter your public Supabase anon  key: ")


async def sign_up(email: str, password: str):
    """Create a new user account."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/signup",
            json={"email": email, "password": password},
            headers={"apiKey": SUPABASE_ANON_KEY},
        )
        assert response.status_code == 200
        return response.json()

## Create two test users
print(f"Creating test users: {email1} and {email2}")
await sign_up(email1, password)
await sign_up(email2, password)
```

Then run the code.

!!! tip "About test emails"
    We'll create two test accounts by adding "+1" and "+2" to your email. For example, if you use "myemail@gmail.com", we'll create "myemail+1@gmail.com" and "myemail+2@gmail.com". All emails will be delivered to your original address.

ÔÜá´©Å Before continuing: Check your email and click both confirmation links. Supabase will will reject `/login` requests until after you have confirmed your users' email.

Now let's test that users can only see their own data. Make sure the server is running (run `langgraph dev`) before proceeding. The following snippet requires the "anon public" key that you copied from the Supabase dashboard while [setting up the auth provider](#setup-auth-provider) previously. 

```python
async def login(email: str, password: str):
    """Get an access token for an existing user."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
            json={
                "email": email,
                "password": password
            },
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Content-Type": "application/json"
            },
        )
        assert response.status_code == 200
        return response.json()["access_token"]


## Log in as user 1
user1_token = await login(email1, password)
user1_client = get_client(
    url="http://localhost:2024", headers={"Authorization": f"Bearer {user1_token}"}
)

## Create a thread as user 1
thread = await user1_client.threads.create()
print(f"Ô£à User 1 created thread: {thread['thread_id']}")

## Try to access without a token
unauthenticated_client = get_client(url="http://localhost:2024")
try:
    await unauthenticated_client.threads.create()
    print("ÔØî Unauthenticated access should fail!")
except Exception as e:
    print("Ô£à Unauthenticated access blocked:", e)

## Try to access user 1's thread as user 2
user2_token = await login(email2, password)
user2_client = get_client(
    url="http://localhost:2024", headers={"Authorization": f"Bearer {user2_token}"}
)

try:
    await user2_client.threads.get(thread["thread_id"])
    print("ÔØî User 2 shouldn't see User 1's thread!")
except Exception as e:
    print("Ô£à User 2 blocked from User 1's thread:", e)
```
The output should look like this:

```shell
Ô£à User 1 created thread: d6af3754-95df-4176-aa10-dbd8dca40f1a
Ô£à Unauthenticated access blocked: Client error '403 Forbidden' for url 'http://localhost:2024/threads'
Ô£à User 2 blocked from User 1's thread: Client error '404 Not Found' for url 'http://localhost:2024/threads/d6af3754-95df-4176-aa10-dbd8dca40f1a'
```

Perfect! Our authentication and authorization are working together:
1. Users must log in to access the bot
2. Each user can only see their own threads

All our users are managed by the Supabase auth provider, so we don't need to implement any additional user management logic.

### Congratulations! ­ƒÄë

You've successfully built a production-ready authentication system for your LangGraph application! Let's review what you've accomplished:

1. Set up an authentication provider (Supabase in this case)
2. Added real user accounts with email/password authentication
3. Integrated JWT token validation into your LangGraph server
4. Implemented proper authorization to ensure users can only access their own data
5. Created a foundation that's ready to handle your next authentication challenge ­ƒÜÇ

This completes our authentication tutorial series. You now have the building blocks for a secure, production-ready LangGraph application.

### What's Next?

Now that you have production authentication, consider:

1. Building a web UI with your preferred framework (see the [Custom Auth](https://github.com/langchain-ai/custom-auth) template for an example)
2. Learn more about the other aspects of authentication and authorization in the [conceptual guide on authentication](#docsconceptsauthmd).
3. Customize your handlers and setup further after reading the [reference docs](#langgraph_sdk.auth.Auth).

<a name="docstutorialsauthgetting_startedmd"></a>

## Setting up Custom Authentication (Part 1/3)

!!! note "This is part 1 of our authentication series:"

    1. Basic Authentication (you are here) - Control who can access your bot
    2. [Resource Authorization](#docstutorialsauthresource_authmd) - Let users have private conversations
    3. [Production Auth](#docstutorialsauthadd_auth_servermd) - Add real user accounts and validate using OAuth2

!!! tip "Prerequisites"

    This guide assumes basic familiarity with the following concepts:

      *  [**Authentication & Access Control**](#docsconceptsauthmd)
      *  [**LangGraph Platform**](#langgraph-platform)

!!! note "Python only"

    We currently only support custom authentication and authorization in Python deployments with `langgraph-api>=0.0.11`. Support for LangGraph.JS will be added soon.

In this tutorial, we will build a chatbot that only lets specific users access it. We'll start with the LangGraph template and add token-based security step by step. By the end, you'll have a working chatbot that checks for valid tokens before allowing access.

### Setting up our project

First, let's create a new chatbot using the LangGraph starter template:

```bash
pip install -U "langgraph-cli[inmem]"
langgraph new --template=new-langgraph-project-python custom-auth
cd custom-auth
```

The template gives us a placeholder LangGraph app. Let's try it out by installing the local dependencies and running the development server.
```shell
pip install -e .
langgraph dev
```
If everything works, the server should start and open the studio in your browser.

> - ­ƒÜÇ API: http://127.0.0.1:2024
> - ­ƒÄ¿ Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
> - ­ƒôÜ API Docs: http://127.0.0.1:2024/docs
> 
> This in-memory server is designed for development and testing.
> For production use, please use LangGraph Cloud.

The graph should run, and if you were to self-host this on the public internet, anyone could access it!

![No auth](./img/no_auth.png)

Now that we've seen the base LangGraph app, let's add authentication to it! 

???+ tip "Placeholder token"
    
    In part 1, we will start with a hard-coded token for illustration purposes.
    We will get to a "production-ready" authentication scheme in part 3, after mastering the basics.


### Adding Authentication

The [`Auth`](#langgraph_sdk.auth.Auth) object lets you register an authentication function that the LangGraph platform will run on every request. This function receives each request and decides whether to accept or reject.

Create a new file `src/security/auth.py`. This is where our code will live to check if users are allowed to access our bot:

```python hl_lines="10 15-16" title="src/security/auth.py"
from langgraph_sdk import Auth

## This is our toy user database. Do not do this in production
VALID_TOKENS = {
    "user1-token": {"id": "user1", "name": "Alice"},
    "user2-token": {"id": "user2", "name": "Bob"},
}

## The "Auth" object is a container that LangGraph will use to mark our authentication function
auth = Auth()


## The `authenticate` decorator tells LangGraph to call this function as middleware
## for every request. This will determine whether the request is allowed or not
@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Check if the user's token is valid."""
    assert authorization
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"
    # Check if token is valid
    if token not in VALID_TOKENS:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")

    # Return user info if valid
    user_data = VALID_TOKENS[token]
    return {
        "identity": user_data["id"],
    }
```

Notice that our [authentication](#langgraph_sdk.auth.Auth.authenticate) handler does two important things:

1. Checks if a valid token is provided in the request's [Authorization header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Authorization)
2. Returns the user's [identity](#langgraph_sdk.auth.types.MinimalUserDict)

Now tell LangGraph to use our authentication by adding the following to the [`langgraph.json`](#configuration-file) configuration:

```json hl_lines="7-9" title="langgraph.json"
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "auth": {
    "path": "src/security/auth.py:auth"
  }
}
```

### Testing Our "Secure" Bot

Let's start the server again to test everything out!

```bash
langgraph dev --no-browser
```

??? note "Custom auth in the studio"

    If you didn't add the `--no-browser`, the studio UI will open in the browser. You may wonder, how is the studio able to still connect to our server? By default, we also permit access from the LangGraph studio, even when using custom auth. This makes it easier to develop and test your bot in the studio. You can remove this alternative authentication option by
    setting `disable_studio_auth: "true"` in your auth configuration:
    ```json
    {
        "auth": {
            "path": "src/security/auth.py:auth",
            "disable_studio_auth": "true"
        }
    }
    ```

Now let's try to chat with our bot. If we've implemented authentication correctly, we should only be able to access the bot if we provide a valid token in the request header. Users will still, however, be able to access each other's resources until we add [resource authorization handlers](#resource-authorization) in the next section of our tutorial.

![Authentication, no authorization handlers](./img/authentication.png)

Run the following code in a file or notebook:

```python
from langgraph_sdk import get_client

## Try without a token (should fail)
client = get_client(url="http://localhost:2024")
try:
    thread = await client.threads.create()
    print("ÔØî Should have failed without token!")
except Exception as e:
    print("Ô£à Correctly blocked access:", e)

## Try with a valid token
client = get_client(
    url="http://localhost:2024", headers={"Authorization": "Bearer user1-token"}
)

## Create a thread and chat
thread = await client.threads.create()
print(f"Ô£à Created thread as Alice: {thread['thread_id']}")

response = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Hello!"}]},
)
print("Ô£à Bot responded:")
print(response)
```

You should see that:

1. Without a valid token, we can't access the bot
2. With a valid token, we can create threads and chat

Congratulations! You've built a chatbot that only lets "authenticated" users access it. While this system doesn't (yet) implement a production-ready security scheme, we've learned the basic mechanics of how to control access to our bot. In the next tutorial, we'll learn how to give each user their own private conversations.

### What's Next?

Now that you can control who accesses your bot, you might want to:

1. Continue the tutorial by going to [Making Conversations Private (Part 2/3)](#docstutorialsauthresource_authmd) to learn about resource authorization.
2. Read more about [authentication concepts](#docsconceptsauthmd).
3. Check out the [API reference](#docscloudreferencesdkpython_sdk_refmd) for more authentication details.

<a name="docstutorialsauthresource_authmd"></a>

## Making Conversations Private (Part 2/3)

!!! note "This is part 2 of our authentication series:" 

    1. [Basic Authentication](#docstutorialsauthgetting_startedmd) - Control who can access your bot 
    2. Resource Authorization (you are here) - Let users have private conversations
    3. [Production Auth](#docstutorialsauthadd_auth_servermd) - Add real user accounts and validate using OAuth2

In this tutorial, we will extend our chatbot to give each user their own private conversations. We'll add [resource-level access control](#resource-level-access-control) so users can only see their own threads.

![Authorization handlers](./img/authorization.png)

???+ tip "Placeholder token"
    
    As we did in [part 1](#docstutorialsauthgetting_startedmd), for this section, we will use a hard-coded token for illustration purposes.
    We will get to a "production-ready" authentication scheme in part 3, after mastering the basics.

### Understanding Resource Authorization

In the last tutorial, we controlled who could access our bot. But right now, any authenticated user can see everyone else's conversations! Let's fix that by adding [resource authorization](#resource-authorization).

First, make sure you have completed the [Basic Authentication](#docstutorialsauthgetting_startedmd) tutorial and that your secure bot can be run without errors:

```bash
cd custom-auth
pip install -e .
langgraph dev --no-browser
```

> - ­ƒÜÇ API: http://127.0.0.1:2024
> - ­ƒÄ¿ Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
> - ­ƒôÜ API Docs: http://127.0.0.1:2024/docs

### Adding Resource Authorization

Recall that in the last tutorial, the [`Auth`](#langgraph_sdk.auth.Auth) object let us register an [authentication function](#authentication), which the LangGraph platform uses to validate the bearer tokens in incoming requests. Now we'll use it to register an **authorization** handler.

Authorization handlers are functions that run **after** authentication succeeds. These handlers can add [metadata](#resource-metadata) to resources (like who owns them) and filter what each user can see.

Let's update our `src/security/auth.py` and add one authorization handler that is run on every request:

```python hl_lines="29-39" title="src/security/auth.py"
from langgraph_sdk import Auth

## Keep our test users from the previous tutorial
VALID_TOKENS = {
    "user1-token": {"id": "user1", "name": "Alice"},
    "user2-token": {"id": "user2", "name": "Bob"},
}

auth = Auth()


@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """Our authentication handler from the previous tutorial."""
    assert authorization
    scheme, token = authorization.split()
    assert scheme.lower() == "bearer"

    if token not in VALID_TOKENS:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")

    user_data = VALID_TOKENS[token]
    return {
        "identity": user_data["id"],
    }


@auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,  # Contains info about the current user
    value: dict,  # The resource being created/accessed
):
    """Make resources private to their creator."""
    # Examples:
    # ctx: AuthContext(
    #     permissions=[],
    #     user=ProxyUser(
    #         identity='user1',
    #         is_authenticated=True,
    #         display_name='user1'
    #     ),
    #     resource='threads',
    #     action='create_run'
    # )
    # value: 
    # {
    #     'thread_id': UUID('1e1b2733-303f-4dcd-9620-02d370287d72'),
    #     'assistant_id': UUID('fe096781-5601-53d2-b2f6-0d3403f7e9ca'),
    #     'run_id': UUID('1efbe268-1627-66d4-aa8d-b956b0f02a41'),
    #     'status': 'pending',
    #     'metadata': {},
    #     'prevent_insert_if_inflight': True,
    #     'multitask_strategy': 'reject',
    #     'if_not_exists': 'reject',
    #     'after_seconds': 0,
    #     'kwargs': {
    #         'input': {'messages': [{'role': 'user', 'content': 'Hello!'}]},
    #         'command': None,
    #         'config': {
    #             'configurable': {
    #                 'langgraph_auth_user': ... Your user object...
    #                 'langgraph_auth_user_id': 'user1'
    #             }
    #         },
    #         'stream_mode': ['values'],
    #         'interrupt_before': None,
    #         'interrupt_after': None,
    #         'webhook': None,
    #         'feedback_keys': None,
    #         'temporary': False,
    #         'subgraphs': False
    #     }
    # }

    # Do 2 things:
    # 1. Add the user's ID to the resource's metadata. Each LangGraph resource has a `metadata` dict that persists with the resource.
    # this metadata is useful for filtering in read and update operations
    # 2. Return a filter that lets users only see their own resources
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)

    # Only let users see their own resources
    return filters
```

The handler receives two parameters:

1. `ctx` ([AuthContext](#langgraph_sdk.auth.types.AuthContext)): contains info about the current `user`, the user's `permissions`, the `resource` ("threads", "crons", "assistants"), and the `action` being taken ("create", "read", "update", "delete", "search", "create_run")
2. `value` (`dict`): data that is being created or accessed. The contents of this dict depend on the resource and action being accessed. See [adding scoped authorization handlers](#scoped-authorization) below for information on how to get more tightly scoped access control.

Notice that our simple handler does two things:

1. Adds the user's ID to the resource's metadata.
2. Returns a metadata filter so users only see resources they own.

### Testing Private Conversations

Let's test our authorization. If we have set things up correctly, we should expect to see all Ô£à messages. Be sure to have your development server running (run `langgraph dev`):

```python
from langgraph_sdk import get_client

## Create clients for both users
alice = get_client(
    url="http://localhost:2024",
    headers={"Authorization": "Bearer user1-token"}
)

bob = get_client(
    url="http://localhost:2024",
    headers={"Authorization": "Bearer user2-token"}
)

## Alice creates an assistant
alice_assistant = await alice.assistants.create()
print(f"Ô£à Alice created assistant: {alice_assistant['assistant_id']}")

## Alice creates a thread and chats
alice_thread = await alice.threads.create()
print(f"Ô£à Alice created thread: {alice_thread['thread_id']}")

await alice.runs.create(
    thread_id=alice_thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Hi, this is Alice's private chat"}]}
)

## Bob tries to access Alice's thread
try:
    await bob.threads.get(alice_thread["thread_id"])
    print("ÔØî Bob shouldn't see Alice's thread!")
except Exception as e:
    print("Ô£à Bob correctly denied access:", e)

## Bob creates his own thread
bob_thread = await bob.threads.create()
await bob.runs.create(
    thread_id=bob_thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Hi, this is Bob's private chat"}]}
)
print(f"Ô£à Bob created his own thread: {bob_thread['thread_id']}")

## List threads - each user only sees their own
alice_threads = await alice.threads.search()
bob_threads = await bob.threads.search()
print(f"Ô£à Alice sees {len(alice_threads)} thread")
print(f"Ô£à Bob sees {len(bob_threads)} thread")
```

Run the test code and you should see output like this:

```bash
Ô£à Alice created assistant: fc50fb08-78da-45a9-93cc-1d3928a3fc37
Ô£à Alice created thread: 533179b7-05bc-4d48-b47a-a83cbdb5781d
Ô£à Bob correctly denied access: Client error '404 Not Found' for url 'http://localhost:2024/threads/533179b7-05bc-4d48-b47a-a83cbdb5781d'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404
Ô£à Bob created his own thread: 437c36ed-dd45-4a1e-b484-28ba6eca8819
Ô£à Alice sees 1 thread
Ô£à Bob sees 1 thread
```

This means:

1. Each user can create and chat in their own threads
2. Users can't see each other's threads
3. Listing threads only shows your own

### Adding scoped authorization handlers {#scoped-authorization}

The broad `@auth.on` handler matches on all [authorization events](#authorization-events). This is concise, but it means the contents of the `value` dict are not well-scoped, and we apply the same user-level access control to every resource. If we want to be more fine-grained, we can also control specific actions on resources.

Update `src/security/auth.py` to add handlers for specific resource types:

```python
## Keep our previous handlers...

from langgraph_sdk import Auth

@auth.on.threads.create
async def on_thread_create(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.create.value,
):
    """Add owner when creating threads.
    
    This handler runs when creating new threads and does two things:
    1. Sets metadata on the thread being created to track ownership
    2. Returns a filter that ensures only the creator can access it
    """
    # Example value:
    #  {'thread_id': UUID('99b045bc-b90b-41a8-b882-dabc541cf740'), 'metadata': {}, 'if_exists': 'raise'}

    # Add owner metadata to the thread being created
    # This metadata is stored with the thread and persists
    metadata = value.setdefault("metadata", {})
    metadata["owner"] = ctx.user.identity
    
    
    # Return filter to restrict access to just the creator
    return {"owner": ctx.user.identity}

@auth.on.threads.read
async def on_thread_read(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.threads.read.value,
):
    """Only let users read their own threads.
    
    This handler runs on read operations. We don't need to set
    metadata since the thread already exists - we just need to
    return a filter to ensure users can only see their own threads.
    """
    return {"owner": ctx.user.identity}

@auth.on.assistants
async def on_assistants(
    ctx: Auth.types.AuthContext,
    value: Auth.types.on.assistants.value,
):
    # For illustration purposes, we will deny all requests
    # that touch the assistants resource
    # Example value:
    # {
    #     'assistant_id': UUID('63ba56c3-b074-4212-96e2-cc333bbc4eb4'),
    #     'graph_id': 'agent',
    #     'config': {},
    #     'metadata': {},
    #     'name': 'Untitled'
    # }
    raise Auth.exceptions.HTTPException(
        status_code=403,
        detail="User lacks the required permissions.",
    )
```

Notice that instead of one global handler, we now have specific handlers for:

1. Creating threads
2. Reading threads
3. Accessing assistants

The first three of these match specific **actions** on each resource (see [resource actions](#resource-actions)), while the last one (`@auth.on.assistants`) matches _any_ action on the `assistants` resource. For each request, LangGraph will run the most specific handler that matches the resource and action being accessed. This means that the four handlers above will run rather than the broadly scoped "`@auth.on`" handler.

Try adding the following test code to your test file:

```python
## ... Same as before
## Try creating an assistant. This should fail
try:
    await alice.assistants.create("agent")
    print("ÔØî Alice shouldn't be able to create assistants!")
except Exception as e:
    print("Ô£à Alice correctly denied access:", e)

## Try searching for assistants. This also should fail
try:
    await alice.assistants.search()
    print("ÔØî Alice shouldn't be able to search assistants!")
except Exception as e:
    print("Ô£à Alice correctly denied access to searching assistants:", e)

## Alice can still create threads
alice_thread = await alice.threads.create()
print(f"Ô£à Alice created thread: {alice_thread['thread_id']}")
```

And then run the test code again:

```bash
Ô£à Alice created thread: dcea5cd8-eb70-4a01-a4b6-643b14e8f754
Ô£à Bob correctly denied access: Client error '404 Not Found' for url 'http://localhost:2024/threads/dcea5cd8-eb70-4a01-a4b6-643b14e8f754'
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/404
Ô£à Bob created his own thread: 400f8d41-e946-429f-8f93-4fe395bc3eed
Ô£à Alice sees 1 thread
Ô£à Bob sees 1 thread
Ô£à Alice correctly denied access:
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500
Ô£à Alice correctly denied access to searching assistants:
```

Congratulations! You've built a chatbot where each user has their own private conversations. While this system uses simple token-based authentication, the authorization patterns we've learned will work with implementing any real authentication system. In the next tutorial, we'll replace our test users with real user accounts using OAuth2.

### What's Next?

Now that you can control access to resources, you might want to:

1. Move on to [Production Auth](#docstutorialsauthadd_auth_servermd) to add real user accounts
2. Read more about [authorization patterns](#authorization)
3. Check out the [API reference](#langgraph_sdk.auth.Auth) for details about the interfaces and methods used in this tutorial


<a name="docstutorialsindexmd"></a>

## Tutorials

New to LangGraph or LLM app development? Read this material to get up and running building your first applications.

### Get Started ­ƒÜÇ {#quick-start}

- [LangGraph Quickstart](introduction.ipynb): Build a chatbot that can use tools and keep track of conversation history. Add human-in-the-loop capabilities and explore how time-travel works.
- [LangGraph Server Quickstart](#docstutorialslanggraph-platformlocal-servermd): Launch a LangGraph server locally and interact with it using REST API and LangGraph Studio Web UI.
- [LangGraph Template Quickstart](#docsconceptstemplate_applicationsmd): Start building with LangGraph Platform using a template application.
- [Deploy with LangGraph Cloud Quickstart](#docscloudquick_startmd): Deploy a LangGraph app using LangGraph Cloud.

### Use cases ­ƒøá´©Å {#use-cases}


Explore practical implementations tailored for specific scenarios:

#### Chatbots

- [Customer Support](customer-support/customer-support.ipynb): Build a multi-functional support bot for flights, hotels, and car rentals.
- [Prompt Generation from User Requirements](chatbots/information-gather-prompting.ipynb): Build an information gathering chatbot.
- [Code Assistant](code_assistant/langgraph_code_assistant.ipynb): Build a code analysis and generation assistant.

#### RAG

- [Agentic RAG](rag/langgraph_agentic_rag.ipynb): Use an agent to figure out how to retrieve the most relevant information before using the retrieved information to answer the user's question.
- [Adaptive RAG](rag/langgraph_adaptive_rag.ipynb): Adaptive RAG is a strategy for RAG that unites (1) query analysis with (2) active / self-corrective RAG. Implementation of: https://arxiv.org/abs/2403.14403
    - For a version that uses a local LLM: [Adaptive RAG using local LLMs](rag/langgraph_adaptive_rag_local.ipynb)
- [Corrective RAG](rag/langgraph_crag.ipynb): Uses an LLM to grade the quality of the retrieved information from the given source, and if the quality is low, it will try to retrieve the information from another source. Implementation of: https://arxiv.org/pdf/2401.15884.pdf 
    - For a version that uses a local LLM: [Corrective RAG using local LLMs](rag/langgraph_crag_local.ipynb)
- [Self-RAG](rag/langgraph_self_rag.ipynb): Self-RAG is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents and generations. Implementation of https://arxiv.org/abs/2310.11511.
    - For a version that uses a local LLM: [Self-RAG using local LLMs](rag/langgraph_self_rag_local.ipynb) 
- [SQL Agent](sql-agent.ipynb): Build a SQL agent that can answer questions about a SQL database.


#### Agent Architectures

##### Multi-Agent Systems

- [Network](multi_agent/multi-agent-collaboration.ipynb): Enable two or more agents to collaborate on a task
- [Supervisor](multi_agent/agent_supervisor.ipynb): Use an LLM to orchestrate and delegate to individual agents
- [Hierarchical Teams](multi_agent/hierarchical_agent_teams.ipynb): Orchestrate nested teams of agents to solve problems
 
##### Planning Agents

- [Plan-and-Execute](plan-and-execute/plan-and-execute.ipynb): Implement a basic planning and execution agent
- [Reasoning without Observation](rewoo/rewoo.ipynb): Reduce re-planning by saving observations as variables
- [LLMCompiler](llm-compiler/LLMCompiler.ipynb): Stream and eagerly execute a DAG of tasks from a planner

##### Reflection & Critique 

- [Basic Reflection](reflection/reflection.ipynb): Prompt the agent to reflect on and revise its outputs
- [Reflexion](reflexion/reflexion.ipynb): Critique missing and superfluous details to guide next steps
- [Tree of Thoughts](tot/tot.ipynb): Search over candidate solutions to a problem using a scored tree
- [Language Agent Tree Search](lats/lats.ipynb): Use reflection and rewards to drive a monte-carlo tree search over agents
- [Self-Discover Agent](self-discover/self-discover.ipynb): Analyze an agent that learns about its own capabilities

#### Evaluation

- [Agent-based](chatbot-simulation-evaluation/agent-simulation-evaluation.ipynb): Evaluate chatbots via simulated user interactions
- [In LangSmith](chatbot-simulation-evaluation/langsmith-agent-simulation-evaluation.ipynb): Evaluate chatbots in LangSmith over a dialog dataset

#### Experimental

- [Web Research (STORM)](storm/storm.ipynb): Generate Wikipedia-like articles via research and multi-perspective QA
- [TNT-LLM](tnt-llm/tnt-llm.ipynb): Build rich, interpretable taxonomies of user intentand using the classification system developed by Microsoft for their Bing Copilot application.
- [Web Navigation](web-navigation/web_voyager.ipynb): Build an agent that can navigate and interact with websites
- [Competitive Programming](usaco/usaco.ipynb): Build an agent with few-shot "episodic memory" and human-in-the-loop collaboration to solve problems from the USA Computing Olympiad; adapted from the ["Can Language Models Solve Olympiad Programming?"](https://arxiv.org/abs/2404.10952v1) paper by Shi, Tang, Narasimhan, and Yao.
- [Complex data extraction](extraction/retries.ipynb): Build an agent that can use function calling to do complex extraction tasks

### LangGraph Platform ­ƒº▒ {#platform}

#### Authentication & Access Control

Add custom authentication and authorization to an existing LangGraph Platform deployment in the following three-part guide:

1. [Setting Up Custom Authentication](#docstutorialsauthgetting_startedmd): Implement OAuth2 authentication to authorize users on your deployment
2. [Resource Authorization](#docstutorialsauthresource_authmd): Let users have private conversations
3. [Connecting an Authentication Provider](#docstutorialsauthadd_auth_servermd): Add real user accounts and validate using OAuth2

<a name="docstutorialslanggraph-platformlocal-servermd"></a>

## QuickStart: Launch Local LangGraph Server

This is a quick start guide to help you get a LangGraph app up and running locally.

!!! info "Requirements"

    - Python >= 3.11
    - [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/): Requires langchain-cli[inmem] >= 0.1.58

### Install the LangGraph CLI

```bash
pip install --upgrade "langgraph-cli[inmem]"
```

### ­ƒî▒ Create a LangGraph App

Create a new app from the `react-agent` template. This template is a simple agent that can be flexibly extended to many tools.

=== "Python Server"

    ```shell
    langgraph new path/to/your/app --template react-agent-python 
    ```

=== "Node Server"

    ```shell
    langgraph new path/to/your/app --template react-agent-js
    ```

!!! tip "Additional Templates"

    If you use `langgraph new` without specifying a template, you will be presented with an interactive menu that will allow you to choose from a list of available templates.

### Install Dependencies

In the root of your new LangGraph app, install the dependencies in `edit` mode so your local changes are used by the server:

```shell
pip install -e .
```

### Create a `.env` file

You will find a `.env.example` in the root of your new LangGraph app. Create
a `.env` file in the root of your new LangGraph app and copy the contents of the `.env.example` file into it, filling in the necessary API keys:

```bash
LANGSMITH_API_KEY=lsv2...
TAVILY_API_KEY=tvly-...
ANTHROPIC_API_KEY=sk-
OPENAI_API_KEY=sk-...
```

??? note "Get API Keys"

    - **LANGSMITH_API_KEY**: Go to the [LangSmith Settings page](https://smith.langchain.com/settings). Then clck **Create API Key**.
    - **ANTHROPIC_API_KEY**: Get an API key from [Anthropic](https://console.anthropic.com/).
    - **OPENAI_API_KEY**: Get an API key from [OpenAI](https://openai.com/).
    - **TAVILY_API_KEY**: Get an API key on the [Tavily website](https://app.tavily.com/).

### ­ƒÜÇ Launch LangGraph Server

```shell
langgraph dev
```

This will start up the LangGraph API server locally. If this runs successfully, you should see something like:

>    Ready!
> 
>    - API: [http://localhost:2024](http://localhost:2024/)
>     
>    - Docs: http://localhost:2024/docs
>     
>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024


!!! note "In-Memory Mode"

    The `langgraph dev` command starts LangGraph Server in an in-memory mode. This mode is suitable for development and testing purposes. For production use, you should deploy LangGraph Server with access to a persistent storage backend.

    If you want to test your application with a persistent storage backend, you can use the `langgraph up` command instead of `langgraph dev`. You will
    need to have `docker` installed on your machine to use this command.

### LangGraph Studio Web UI

LangGraph Studio Web is a specialized UI that you can connect to LangGraph API server to enable visualization, interaction, and debugging of your application locally. Test your graph in the LangGraph Studio Web UI by visiting the URL provided in the output of the `langgraph dev` command.

>    - LangGraph Studio Web UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

!!! info "Connecting to a server with a custom host/port"

    If you are running the LangGraph API server with a custom host / port, you can point the Studio Web UI at it by changing the `baseUrl` URL param. For example, if you are running your server on port 8000, you can change the above URL to the following:

    ```
    https://smith.langchain.com/studio/baseUrl=http://127.0.0.1:8000
    ```


!!! warning "Safari Compatibility"
    
    Currently, LangGraph Studio Web does not support Safari when running a server locally.

### Test the API

=== "Python SDK (Async)"

    **Install the LangGraph Python SDK**

    ```shell
    pip install langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```python
    from langgraph_sdk import get_client

    client = get_client(url="http://localhost:2024")

    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Python SDK (Sync)"

    **Install the LangGraph Python SDK**

    ```shell
    pip install langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```python
    from langgraph_sdk import get_sync_client

    client = get_sync_client(url="http://localhost:2024")

    for chunk in client.runs.stream(
        None,  # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    ```

=== "Javascript SDK"

    **Install the LangGraph JS SDK**

    ```shell
    npm install @langchain/langgraph-sdk
    ```

    **Send a message to the assistant (threadless run)**

    ```js
    const { Client } = await import("@langchain/langgraph-sdk");

    // only set the apiUrl if you changed the default port when calling langgraph dev
    const client = new Client({ apiUrl: "http://localhost:2024"});

    const streamResponse = client.runs.stream(
        null, // Threadless run
        "agent", // Assistant ID
        {
            input: {
                "messages": [
                    { "role": "user", "content": "What is LangGraph?"}
                ]
            },
            streamMode: "messages",
        }
    );

    for await (const chunk of streamResponse) {
        console.log(`Receiving new event of type: ${chunk.event}...`);
        console.log(JSON.stringify(chunk.data));
        console.log("\n\n");
    }
    ```

=== "Rest API"

    ```bash
    curl -s --request POST \
        --url "http://localhost:2024/runs/stream" \
        --header 'Content-Type: application/json' \
        --data "{
            \"assistant_id\": \"agent\",
            \"input\": {
                \"messages\": [
                    {
                        \"role\": \"human\",
                        \"content\": \"What is LangGraph?\"
                    }
                ]
            },
            \"stream_mode\": \"updates\"
        }" 
    ```

!!! tip "Auth"

    If you're connecting to a remote server, you will need to provide a LangSmith
    API Key for authorization. Please see the API Reference for the clients
    for more information.

### Next Steps

Now that you have a LangGraph app running locally, take your journey further by exploring deployment and advanced features:

#### ­ƒîÉ Deploy to LangGraph Cloud

- **[LangGraph Cloud Quickstart](#docscloudquick_startmd)**: Deploy your LangGraph app using LangGraph Cloud.

#### ­ƒôÜ Learn More about LangGraph Platform

Expand your knowledge with these resources:

- **[LangGraph Platform Concepts](#langgraph-platform)**: Understand the foundational concepts of the LangGraph Platform.  
- **[LangGraph Platform How-to Guides](#langgraph-platform)**: Discover step-by-step guides to build and deploy applications.

#### ­ƒøá´©Å Developer References

Access detailed documentation for development and API usage:

- **[LangGraph Server API Reference](../../cloud/reference/api/api_ref.html)**: Explore the LangGraph Server API documentation.  
- **[Python SDK Reference](#docscloudreferencesdkpython_sdk_refmd)**: Explore the Python SDK API Reference.
- **[JS/TS SDK Reference](../../cloud/reference/sdk/js_ts_sdk_ref.md)**: Explore the Python SDK API Reference.
