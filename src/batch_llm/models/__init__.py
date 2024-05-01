from batch_llm.models.azure_openai.azure_openai import AsyncAzureOpenAIModel
from batch_llm.models.base import AsyncBaseModel
from batch_llm.models.gemini.gemini import AsyncGeminiModel
from batch_llm.models.huggingface_tgi.huggingface_tgi import AsyncHuggingfaceTGIModel
from batch_llm.models.ollama.ollama import AsyncOllamaModel
from batch_llm.models.openai.openai import AsyncOpenAIModel
from batch_llm.models.testing.testing_model import AsyncTestModel

ASYNC_MODELS: dict[str, AsyncBaseModel] = {
    "azure-openai": AsyncAzureOpenAIModel,
    "gemini": AsyncGeminiModel,
    "openai": AsyncOpenAIModel,
    "test": AsyncTestModel,
    "ollama": AsyncOllamaModel,
    "huggingface-tgi": AsyncHuggingfaceTGIModel,
}

__all__ = ["MODELS", "ASYNC_MODELS"]
