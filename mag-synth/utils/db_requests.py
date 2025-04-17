import os
from loguru import logger
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from .schemas import (
    PropertiesFilter,
    PropertiesFilterMode,
)


EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "thenlper/gte-large")
EMBEDDING_DIMENSIONS = os.getenv("EMBEDDING_DIMENSIONS", 1024)

# load model on the startup
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_ID)


def connect_to_es() -> Elasticsearch:
    es_client = Elasticsearch(
        hosts=[os.environ["ES_HOST"]],
        api_key=os.environ["ES_API_KEY"],
    )
    try:
        info = es_client.info()
        logger.info(f"Connected to: {info['cluster_name']}")
    except Exception as e:
        err_text = str(e)
        logger.error(err_text)
        raise RuntimeError("Cannot connect to ES cluster:" + err_text)
    return es_client


def get_embedding(text: str) -> list[float]:
    if not text.strip():
        logger.error("Attempted to get embedding for empty text.")
        return []
    embedding = EMBEDDING_MODEL.encode(text)
    return embedding.tolist()


def vector_search(
    client: Elasticsearch,
    index_name: str,
    query: str,
) -> list[dict[str, str | float]]:
    question = get_embedding(query)
    body = {
        "size": 5,
        "knn": {
            "field": "embedding",
            "query_vector": question,
            "k": 10,
            "num_candidates": 150,
        }
    }
    examples = client.search(
        index=index_name,
        body=body,
    )
    res = []
    for item in examples["hits"]["hits"]:
        res.append(
            {k: v for k, v in item["_source"].items() if k != "embedding"})
    return res


def get_db_records(query: str):
    es_client = connect_to_es()

    samples = vector_search(
        es_client,
        index_name=os.getenv("ES_INDEX_NAME", "synthesis"),
        query=query,
    )
    return samples


def search_db_by_properties(properties: list[PropertiesFilter]) -> list[dict[str, str | int | float]]:
    es_client = connect_to_es()

    properties_filter = []
    for prop in properties:
        prop_filter = {
            "range": {
                prop.property_name: {}
            }
        }
        match prop.mode:
            case PropertiesFilterMode.both:
                prop_filter["range"][prop.property_name]["gte"] = prop.greater_than
                prop_filter["range"][prop.property_name]["lte"] = prop.less_than
            case PropertiesFilterMode.gt:
                prop_filter["range"][prop.property_name]["gte"] = prop.greater_than
            case PropertiesFilterMode.lt:
                prop_filter["range"][prop.property_name]["lte"] = prop.less_than
        properties_filter.append(prop_filter)
    

    body = {
        "query": {
            "bool": {
                "must": properties_filter
            }
        }
    }

    response = es_client.search(
        index=os.getenv("ES_INDEX_NAME", "synthesis"),
        body=body,
    )
    res = []
    for item in response["hits"]["hits"]:
        res.append(
            {k: v for k, v in item["_source"].items() if k != "embedding"})
    return res
