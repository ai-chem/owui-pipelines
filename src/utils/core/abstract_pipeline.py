"""
title: Abstract OWUI Pipeline
description: An abstract class documenting necessary interface for creating OWUI pipelines 
"""

from abc import ABC, abstractmethod
from typing import (
    Generator,
    Iterator,
)


class AbstractPipeline(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def on_startup(self):
        pass

    @abstractmethod
    async def on_shutdown(self):
        pass
    
    @abstractmethod
    def pipe(
            self,
            user_message: str,
            model_id: str,
            messages: list[dict],
            body: dict,
    ) -> str | Generator | Iterator:
        pass

