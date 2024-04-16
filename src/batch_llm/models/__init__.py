from batch_llm.models.azure_openai.azure_openai import (
    AsyncAzureOpenAI,
    AzureOpenAIModel,
)
from batch_llm.models.gemini.gemini import AsyncGeminiModel, GeminiModel
from batch_llm.models.testing.testing_model import TestModel

MODELS = {
    "test": TestModel,
    "azure_openai": AzureOpenAIModel,
    "gemini": GeminiModel,
}

ASYNC_MODELS = {
    "azure_openai": AsyncAzureOpenAI,
    "gemini": AsyncGeminiModel,
}

__all__ = ["MODELS", "ASYNC_MODELS"]
