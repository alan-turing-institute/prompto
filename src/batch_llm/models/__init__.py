from batch_llm.models.azure_openai.azure_openai import (
    AsyncAzureOpenAIModel,
    AzureOpenAIModel,
)
from batch_llm.models.base import AsyncBaseModel, BaseModel
from batch_llm.models.gemini.gemini import AsyncGeminiModel, GeminiModel
from batch_llm.models.huggingface_tgi.huggingface_tgi import AsyncHuggingfaceTGIModel
from batch_llm.models.ollama.ollama import AsyncOllamaModel
from batch_llm.models.openai.openai import AsyncOpenAIModel, OpenAIModel
from batch_llm.models.testing.testing_model import AsyncTestModel, TestModel

MODELS: dict[str, BaseModel] = {
    "azure-openai": AzureOpenAIModel,
    "gemini": GeminiModel,
    "openai": OpenAIModel,
    "test": TestModel,
}

ASYNC_MODELS: dict[str, AsyncBaseModel] = {
    "azure-openai": AsyncAzureOpenAIModel,
    "gemini": AsyncGeminiModel,
    "openai": AsyncOpenAIModel,
    "test": AsyncTestModel,
    "ollama": AsyncOllamaModel,
    "huggingface-tgi": AsyncHuggingfaceTGIModel,
}

__all__ = ["MODELS", "ASYNC_MODELS"]
