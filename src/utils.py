import os
from pydantic import BaseModel
from dotenv import load_dotenv
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex

from src.models import llm_factory

load_dotenv()

class Neo4JCredentialsManager(BaseModel):
    username: str
    password: str
    url: str
    database: str


def _get_credentials(
        username: str = os.environ.get("NEO_USERNAME"),
        password: str = os.environ.get("NEO_PASSWORD"),
        url: str = os.environ.get("NEO_URL"),
        database: str = os.environ.get("NEO_DATABASE"),
):
    """
    Currently uses the credentials set in env vars to create credentials for access to the Neo4J database.
    For fine-grained controls, please see: https://neo4j.com/docs/operations-manual/current/tutorial/access-control/
    """
    return Neo4JCredentialsManager(
        username=username,
        password=password,
        url=url,
        database=database,
    )

def graph_store_factory():
    credentials = _get_credentials()
    return Neo4jPropertyGraphStore(
        username=credentials.username,
        password=credentials.password,
        url=credentials.url,
        database=credentials.database,
    )

def get_index()->PropertyGraphIndex:
    graph_store = graph_store_factory()
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        llm=llm_factory(),
        use_async=False,
        show_progress=True,
    )
    return index