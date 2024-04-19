from batch_llm.models.azure_openai.azure_openai import (
    AsyncAzureOpenAIModel,
    AzureOpenAIModel,
)
from batch_llm.models.base import AsyncBaseModel, BaseModel
from batch_llm.models.gemini.gemini import AsyncGeminiModel, GeminiModel
from batch_llm.models.ollama.ollama import AsyncOllamaModel
from batch_llm.models.testing.testing_model import AsyncTestModel, TestModel

MODELS: dict[str, BaseModel] = {
    "azure_openai": AzureOpenAIModel,
    "gemini": GeminiModel,
    "test": TestModel,
}

ASYNC_MODELS: dict[str, AsyncBaseModel] = {
    "azure_openai": AsyncAzureOpenAIModel,
    "gemini": AsyncGeminiModel,
    "test": AsyncTestModel,
    "ollama": AsyncOllamaModel,
}

__all__ = ["MODELS", "ASYNC_MODELS"]
