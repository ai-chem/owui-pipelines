import os
import asyncio
from typing import Optional

import dotenv
from loguru import logger
from pydantic import BaseModel

import torch
import joblib
import pandas as pd
from openai import OpenAI
from catboost import CatBoostRegressor
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from pipelines.utils.core.schemas import Message
from pipelines.utils.magsynth.schemas import (
    CategorizationResult,
    PropertiesFilter,
    PropertiesFilterMode,
    QueryCategory
)
from pipelines.utils.magsynth.prompts import (
    query_categorization,
    general_query_answer,
)

dotenv.load_dotenv('pipelines/magsynth-dev.env')


class Pipeline:
    class Valves(BaseModel):
        GOOGLE_AI_API: Optional[str]
        GOOGLE_AI_API_KEY: Optional[str]
        QUERY_CATEGORIZATION_MODEL_ID: Optional[str]
        GENERAL_QUERY_MODEL_ID: Optional[str]
        SYNTHESIS_QUERY_MODEL_ID: Optional[str]
        PROPERTIES_QUERY_MODEL_ID: Optional[str]
        REGULAR_EMBEDDING_MODEL_ID: Optional[str]
        REDUCED_EMBEDDING_MODEL_ID: Optional[str]
        REGULAR_EMBEDDING_DIMENSIONS: Optional[int]
        REDUCED_EMBEDDING_DIMENSIONS: Optional[int]
        ES_HOST: Optional[str]
        ES_API_KEY: Optional[str]
        ES_VECTOR_SEARCH_INDEX: Optional[str]

    def __init__(self):
        self.valves = self.Valves(
            **{
                "GOOGLE_AI_API": os.getenv("GOOGLE_AI_API"),
                "GOOGLE_AI_API_KEY": os.getenv("GOOGLE_AI_API_KEY"),
                "QUERY_CATEGORIZATION_MODEL_ID": os.getenv("QUERY_CATEGORIZATION_MODEL_ID", "gemini-2.0-flash"),
                "GENERAL_QUERY_MODEL_ID": os.getenv("GENERAL_QUERY_MODEL_ID", "gemini-2.0-flash"),
                "SYNTHESIS_QUERY_MODEL_ID": os.getenv("SYNTHESIS_QUERY_MODEL_ID", "gemini-2.0-flash"),
                "PROPERTIES_QUERY_MODEL_ID": os.getenv("PROPERTIES_QUERY_MODEL_ID", "gemini-2.0-flash"),
                "REGULAR_EMBEDDING_MODEL_ID": os.getenv("REGULAR_EMBEDDING_MODEL_ID", "thenlper/gte-large"),
                "REDUCED_EMBEDDING_MODEL_ID": os.getenv("REDUCED_EMBEDDING_MODEL_ID", "thenlper/gte-small"),
                "REGULAR_EMBEDDING_DIMENSIONS": int(os.getenv("REGULAR_EMBEDDING_DIMENSIONS", 1024)),
                "REDUCED_EMBEDDING_DIMENSIONS": int(os.getenv("REDUCED_EMBEDDING_DIMENSIONS", 384)),
                "ES_HOST": os.getenv("ES_HOST"),
                "ES_API_KEY": os.getenv("ES_API_KEY"),
                "ES_VECTOR_SEARCH_INDEX": os.getenv("ES_VECTOR_SEARCH_INDEX", "magnetic-extended"),
            }
        )
        self.openai_client: OpenAI | None = None
        self.es_client: Elasticsearch | None = None

    def __connect_to_es(self) -> None:
        if self.valves.ES_HOST is None:
            logger.error(f"Elasticsearch host not set")
            return

        if self.valves.ES_API_KEY is None:
            logger.error(f"Elasticsearch API key not set")
            return

        self.es_client = Elasticsearch(
            hosts=[self.valves.ES_HOST],
            api_key=self.valves.ES_API_KEY,
        )
        try:
            info = self.es_client.info()
            logger.info(f"Connected to: {info['cluster_name']}")
        except Exception as e:
            err_text = str(e)
            logger.error("Cannot connect to ES cluster:" + err_text)

    def __connect_to_openai(self):
        if self.valves.GOOGLE_AI_API is None:
            logger.error(f"OpenAI API URL not set")
            return

        if self.valves.GOOGLE_AI_API_KEY is None:
            logger.error(f"OpenAI API key not set")
            return

        self.openai_client = OpenAI(
            base_url=self.valves.GOOGLE_AI_API,
            api_key=self.valves.GOOGLE_AI_API_KEY,
        )

    async def __setup_embedding_models(self):
        self.regular_embedding_model = SentenceTransformer(self.valves.REGULAR_EMBEDDING_MODEL_ID)
        self.reduced_embedding_model = SentenceTransformer(self.valves.REDUCED_EMBEDDING_MODEL_ID)

    async def __setup_predictive_models(self):
        weights_path = "pipelines/utils/magsynth/weights"

        self.features = joblib.load(f"{weights_path}/features.pkl")

        self.coer_model = joblib.load(f"{weights_path}/catboost_model_coer.pkl")
        self.coer_scaler = joblib.load(f"{weights_path}/scaler_coer.pkl")

        # self.mr_model = joblib.load(f"{weights_path}/catboost_model_mr.pkl")
        # self.mr_scaler = joblib.load(f"{weights_path}/scaler_mr.pkl")

        self.sat_model = joblib.load(f"{weights_path}/catboost_model_sat.pkl")
        self.sat_scaler = joblib.load(f"{weights_path}/scaler_sat.pkl")

    def __check_connections(self) -> bool:
        if self.openai_client is None:
            self.__connect_to_openai()

        if self.openai_client is None:
            logger.error("Cannot connect to OpenAI")
            return False

        if self.es_client is None:
            self.__connect_to_es()

        if self.es_client is None:
            logger.error("Cannot connect to Elasticsearch")
            return False

        return True

    def __get_text_embedding(self, text: str) -> list[float]:
        if not text.strip():
            logger.error("Attempted to get embedding for empty text.")
            return []
        embedding = self.regular_embedding_model.encode(text)
        return embedding.tolist()

    def __get_reduced_embedding(self, text: str) -> list[float]:
        if not text.strip():
            logger.error("Attempted to get embedding for empty text.")
            return []
        embedding = self.reduced_embedding_model.encode(text)
        return embedding.tolist()

    def __predict(self, items: list[dict[str, str | float | int]]) -> list[dict[str, str | float | int]]:
        df = pd.DataFrame(items)

        coer_scaled_item = self.coer_scaler.transform(df[self.features])
        df["coer_prediction"] = self.coer_model.predict(coer_scaled_item)

        # mr_scaled_item = self.mr_scaler.transform(df[self.features])
        # df["mr_prediction"] = self.mr_model.predict(mr_scaled_item)

        sat_scaled_item = self.sat_scaler.transform(df[self.features])
        df["sat_prediction"] = self.sat_model.predict(sat_scaled_item)

        return df.to_dict(orient="records")

    def __vector_search_by_query(self, query: str) -> list[dict[str, str | float]]:
        question = self.__get_text_embedding(query)
        body = {
            "size": 5,
            "knn": {
                "field": "synthesis_embedding",
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
                {k: v for k, v in item["_source"].items() if "embedding" not in k})

        return res

    def __vector_search_by_properties(self, properties: list[PropertiesFilter]) -> list[dict[str, str | int | float]]:
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

    def __categorize_message(self, user_message: str) -> CategorizationResult | None:
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

        return response.choices[0].message.content

    def __answer_synthesis_query(self, query: str, messages: list[dict]) -> None:
        system_message = Message(
            role="system",
            content=general_query_answer.system_prompt,
        ).model_dump()

        db_examples = self.__vector_search_by_query(query)

        db_examples = self.__predict(db_examples)

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

        return response.choices[0].message.content

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

        db_records = self.__predict(db_records)

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

        return response.choices[0].message.content

    async def on_startup(self):
        await self.__setup_embedding_models()
        await self.__setup_predictive_models()
        self.__connect_to_openai()
        self.__connect_to_es()

    async def on_shutdown(self):
        self.openai_client.close()
        self.es_client.close()
        del self.regular_embedding_model
        torch.cuda.empty_cache()

    def pipe(self, user_message, model_id, messages, body):
        all_connected = self.__check_connections()
        if not all_connected:
            return "Cannot answer the message due to connection errors. Reach the administator for help."

        message_category = self.__categorize_message(user_message)
        match message_category.category:
            case QueryCategory.general:
                return self.__answer_general_query(user_message, messages)
            case QueryCategory.synthesis:
                return self.__answer_synthesis_query(user_message, messages)
            case QueryCategory.properties:
                return self.__answer_properties_query(
                    user_message,
                    message_category.get_transformed_content(),
                    messages,
                )
            case _:
                return "Unknown message category: " + message_category.category.value

async def main():
    pipeline = Pipeline()
    await pipeline.on_startup()
    response = pipeline.pipe(
        "What is the saturation magnetization parameter of the LiFePO4?",
        "gemini-2.0-flash",
        [],
        {},
    )
    print(response)
    await pipeline.on_shutdown()


if __name__ == "__main__":
    asyncio.run(main())
