from prompto.models.azure_openai.azure_openai import AsyncAzureOpenAIModel
from prompto.models.base import AsyncBaseModel
from prompto.models.gemini.gemini import AsyncGeminiModel
from prompto.models.huggingface_tgi.huggingface_tgi import AsyncHuggingfaceTGIModel
from prompto.models.ollama.ollama import AsyncOllamaModel
from prompto.models.openai.openai import AsyncOpenAIModel
from prompto.models.quart.quart import AsyncQuartModel
from prompto.models.testing.testing_model import AsyncTestModel

ASYNC_MODELS: dict[str, AsyncBaseModel] = {
    "azure-openai": AsyncAzureOpenAIModel,
    "gemini": AsyncGeminiModel,
    "openai": AsyncOpenAIModel,
    "test": AsyncTestModel,
    "ollama": AsyncOllamaModel,
    "huggingface-tgi": AsyncHuggingfaceTGIModel,
    "quart": AsyncQuartModel,
}

__all__ = ["MODELS", "ASYNC_MODELS"]
