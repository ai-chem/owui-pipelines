import enum
from typing import Literal
from pydantic import BaseModel


class LLMProviderEnum(str, enum.Enum):
    GOOGLE = "GOOGLE"


class LLMProvideAPI(str, enum.Enum):
    GOOGLE = "{base_url}/google/v1beta/openai/"


class Message(BaseModel):
    role: Literal["user", "system"]
    content: str