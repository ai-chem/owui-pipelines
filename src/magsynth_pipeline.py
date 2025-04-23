import os
from typing import Optional

import torch
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from backend.src.pipeline.schemas import QueryCategory
from utils.core.schemas import Message
from utils.magsynth.schemas import (
    CategorizationResult,
    PropertiesFilter,
    PropertiesFilterMode,
)
from utils.magsynth.prompts import (
    query_categorization,
    general_query_answer,
)


class Pipeline:
    class Valves(BaseModel):
        GOOGLE_AI_API: Optional[str]
        GOOGLE_AI_API_KEY: Optional[str]
        QUERY_CATEGORIZATION_MODEL_ID: Optional[str]
        GENERAL_QUERY_MODEL_ID: Optional[str]
        SYNTHESIS_QUERY_MODEL_ID: Optional[str]
        PROPERTIES_QUERY_MODEL_ID: Optional[str]
        EMBEDDING_MODEL_ID: Optional[str]
        EMBEDDING_DIMENSIONS: Optional[int]
        ES_HOST: Optional[str]
        ES_API_KEY: Optional[str]
        ES_VECTOR_SEARCH_INDEX: Optional[str]

    def __init__(self):
        self.valves = self.Valves(
            **{
                "GOOGLE_AI_API": os.getenv("GOOGLE_AI_API"),
                "GOOGLE_AI_API_KEY": os.getenv("GOOGLE_AI_API_KEY"),
                "QUERY_CATEGORIZATION_MODEL_ID": os.getenv("QUERY_CATEGORIZATION_MODEL_ID", "gemini-flash-2.0"),
                "GENERAL_QUERY_MODEL_ID": os.getenv("GENERAL_QUERY_MODEL_ID", "gemini-flash-2.0"),
                "SYNTHESIS_QUERY_MODEL_ID": os.getenv("SYNTHESIS_QUERY_MODEL_ID", "gemini-flash-2.0"),
                "PROPERTIES_QUERY_MODEL_ID": os.getenv("PROPERTIES_QUERY_MODEL_ID", "gemini-flash-2.0"),
                "EMBEDDING_MODEL_ID": os.getenv("EMBEDDING_MODEL_ID", "thenlper/gte-large"),
                "EMBEDDING_DIMENSIONS": int(os.getenv("EMBEDDING_DIMENSIONS", 1024)),
                "ES_HOST": os.getenv("ES_HOST"),
                "ES_API_KEY": os.getenv("ES_API_KEY"),
                "ES_VECTOR_SEARCH_INDEX": os.getenv("ES_VECTOR_SEARCH_INDEX", "magnetic"),
            }
        )

    async def __connect_to_es(self) -> None:
        self.es_client: Elasticsearch | None = None

        if self.valves.ES_HOST is None:
            logger.error(f"Elasticsearch host not set")
            return

        if self.valves.ES_API_KEY is None:
            logger.error(f"Elasticsearch API key not set")
            return

        self.es_client = Elasticsearch(
            hosts=[os.environ["ES_HOST"]],
            api_key=os.environ["ES_API_KEY"],
        )
        try:
            info = self.es_client.info()
            logger.info(f"Connected to: {info['cluster_name']}")
        except Exception as e:
            err_text = str(e)
            logger.error("Cannot connect to ES cluster:" + err_text)

    async def __connect_to_openai(self):
        self.openai_client = OpenAI(
            base_url=self.valves.GOOGLE_AI_API,
            api_key=self.valves.GOOGLE_AI_API_KEY,
        )

    async def __setup_embedding_model(self):
        self.embedding_model = SentenceTransformer(self.valves.EMBEDDING_MODEL_ID)

    def __get_text_embedding(self, text: str) -> list[float]:
        if not text.strip():
            logger.error("Attempted to get embedding for empty text.")
            return []
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def __vector_search_by_query(self, query: str) -> list[dict[str, str | float]]:
        if self.es_client is None:
            self.__connect_to_es()

        if self.es_client is None:
            raise RuntimeError("Cannot connect to Elasticsearch")

        question = self.__get_text_embedding(query)
        body = {
            "size": 5,
            "knn": {
                "field": "embedding",
                "query_vector": question,
                "k": 10,
                "num_candidates": 150,
            }
        }
        examples = self.es_client.search(
            index=self.valves.ES_VECTOR_SEARCH_INDEX,
            body=body,
        )
        res = []
        for item in examples["hits"]["hits"]:
            res.append(
                {k: v for k, v in item["_source"].items() if k != "embedding"})
        return res

    def __vector_search_by_properties(self, properties: list[PropertiesFilter]) -> list[dict[str, str | int | float]]:
        if self.es_client is None:
            self.__connect_to_es()

        if self.es_client is None:
            raise RuntimeError("Cannot connect to Elasticsearch")

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

        response = self.es_client.search(
            index=self.valves.ES_VECTOR_SEARCH_INDEX,
            body=body,
        )
        res = []
        for item in response["hits"]["hits"]:
            res.append(
                {k: v for k, v in item["_source"].items() if k != "embedding"})
        return res

    def __categorize_message(self, user_message: str) -> CategorizationResult:
        completion = self.openai_client.beta.chat.completions.parse(
            model=self.valves.QUERY_CATEGORIZATION_MODEL_ID,
            messages=[
                Message(role="system", content=query_categorization.system_prompt).model_dump(),
                Message(role="user", content=query_categorization.query_prompt.format(query=user_message)).model_dump(),
            ],
            response_format=CategorizationResult
        )

        return completion.choices[0].message.parsed

    def __answer_general_query(self, query: str, messages: list[dict]):
        system_message = Message(
            role="system",
            content=general_query_answer.system_prompt,
        ).model_dump()

        user_message = Message(
            role="user",
            content=general_query_answer.general_user_prompt.format(query=query)
        ).model_dump()

        local_messages = [system_message] + messages + [user_message]

        response = self.openai_client.chat.completions.create(
            model=self.valves.GENERAL_QUERY_MODEL_ID,
            n=1,
            messages=local_messages,
        )

        return response

    def __answer_synthesis_query(self, query: str, messages: list[dict]):
        system_message = Message(
            role="system",
            content=general_query_answer.system_prompt,
        ).model_dump()

        db_examples = self.__vector_search_by_query(query)

        user_message = Message(
            role="user",
            content=general_query_answer.synthesis_user_prompt.format(
                query=query,
                examples=db_examples,
            )
        ).model_dump()

        local_messages = [system_message] + messages + [user_message]

        response = self.openai_client.chat.completions.create(
            model=self.valves.GENERAL_QUERY_MODEL_ID,
            n=1,
            messages=local_messages,
        )

        return response

    def __answer_properties_query(
            self,
            query: str,
            properties_content: list[dict[str, str | float | int | None]],
            messages: list[dict]
    ):
        system_message = Message(
            role="system",
            content=general_query_answer.system_prompt,
        ).model_dump()

        parsed_properties_content = [PropertiesFilter(**item) for item in properties_content]
        db_records = self.__vector_search_by_properties(parsed_properties_content)

        user_message = Message(
            role="user",
            content=general_query_answer.db_summary_user_prompt.format(
                query=query,
                records=db_records,
            )
        ).model_dump()

        local_messages = [system_message] + messages + [user_message]

        response = self.openai_client.chat.completions.create(
            model=self.valves.GENERAL_QUERY_MODEL_ID,
            n=1,
            messages=local_messages,
        )

        return response

    async def on_startup(self):
        await self.__connect_to_openai()
        await self.__connect_to_es()
        await self.__setup_embedding_model()

    async def on_shutdown(self):
        self.openai_client.close()
        self.es_client.close()
        del self.embedding_model
        torch.cuda.empty_cache()

    def pipe(self, user_message, model_id, messages, body):
        message_category = self.__categorize_message(user_message)
        match message_category.category:
            case QueryCategory.general:
                return self.__answer_general_query(user_message, messages)
            case QueryCategory.synthesis:
                return self.__answer_synthesis_query(user_message, messages)
            case QueryCategory.properties:
                return self.__answer_properties_query(
                    user_message,
                    message_category.content,
                    messages,
                )
            case _:
                return "Unknown message category: " + message_category.category.value
