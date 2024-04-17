from batch_llm.models.azure_openai.azure_openai import (
    AsyncAzureOpenAIModel,
    AzureOpenAIModel,
)
from batch_llm.models.gemini.gemini import AsyncGeminiModel, GeminiModel
from batch_llm.models.testing.testing_model import AsyncTestModel, TestModel

MODELS = {
    "azure_openai": AzureOpenAIModel,
    "gemini": GeminiModel,
    "test": TestModel,
}

ASYNC_MODELS = {
    "azure_openai": AsyncAzureOpenAIModel,
    "gemini": AsyncGeminiModel,
    "test": AsyncTestModel,
}

__all__ = ["MODELS", "ASYNC_MODELS"]
