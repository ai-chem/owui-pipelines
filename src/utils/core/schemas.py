import enum
from typing import Literal
from pydantic import BaseModel


class LLMProviderEnum(enum.Enum, str):
    GOOGLE = "GOOGLE"


class LLMProvideAPI(enum.Enum, str):
    GOOGLE = "{base_url}/google/v1beta/openai/"


class Message(BaseModel):
    role: Literal["user", "system"]
    content: str