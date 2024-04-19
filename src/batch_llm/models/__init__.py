from batch_llm.models.azure_openai.azure_openai import (
    AsyncAzureOpenAIModel,
    AzureOpenAIModel,
)
from batch_llm.models.base import AsyncBaseModel, BaseModel
from batch_llm.models.gemini.gemini import AsyncGeminiModel, GeminiModel
from batch_llm.models.openai.openai import AsyncOpenAIModel, OpenAIModel
from batch_llm.models.testing.testing_model import AsyncTestModel, TestModel

MODELS: dict[str, BaseModel] = {
    "azure_openai": AzureOpenAIModel,
    "gemini": GeminiModel,
    "openai": OpenAIModel,
    "test": TestModel,
}

ASYNC_MODELS: dict[str, AsyncBaseModel] = {
    "azure_openai": AsyncAzureOpenAIModel,
    "gemini": AsyncGeminiModel,
    "openai": AsyncOpenAIModel,
    "test": AsyncTestModel,
}

__all__ = ["MODELS", "ASYNC_MODELS"]
