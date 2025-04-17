import os
from openai import OpenAI
from utils.schemas import (
    Message,
    CategorizationResult,
)
from prompts import system_prompt, query_prompt


GOOGLE_AI_API_URL = os.environ["GEMINI_API_URL"]
GOOGLE_AI_API_KEY = os.environ["GOOGLE_AI_API_KEY"]


async def categorize_query(query: str, model_id: str) -> CategorizationResult:
    client = OpenAI(
        api_key=GOOGLE_AI_API_KEY,
        base_url=GOOGLE_AI_API_URL,
    )

    completion = client.beta.chat.completions.parse(
        model=model_id,
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=query_prompt.format(query=query))
        ],
        response_format=CategorizationResult
    )

    return completion.choices[0].message.parsed
