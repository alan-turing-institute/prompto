import warnings

from prompto.apis.base import AsyncAPI
from prompto.apis.testing import TestAPI


class DependencyWarning(Warning):
    pass


# define the API names that are available in the pipeline and the corresponding model classes
ASYNC_APIS: dict[str, AsyncAPI] = {}

# import the model classes - if the import fails (i.e. optional dependencies were not installed),
# the model will not be available

# test model is always available
ASYNC_APIS["test"] = TestAPI

# import the other APIs if the dependencies are available
try:
    from prompto.apis.azure_openai import AzureOpenAIAPI

    ASYNC_APIS["azure-openai"] = AzureOpenAIAPI
except ImportError as exc:
    warnings.warn(
        message=(
            f"Azure OpenAI model ('azure-openai') not available. Perhaps you need to install the Azure OpenAI dependencies: {exc}. "
            "Try `pip install prompto[azure_openai]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.openai import OpenAIAPI

    ASYNC_APIS["openai"] = OpenAIAPI
except ImportError as exc:
    warnings.warn(
        message=(
            f"OpenAI model ('openai') not available. Perhaps you need to install the OpenAI dependencies: {exc}. "
            "Try `pip install prompto[openai]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.gemini import GeminiAPI

    ASYNC_APIS["gemini"] = GeminiAPI
except ImportError as exc:
    warnings.warn(
        message=(
            f"Gemini API ('gemini') not available. Perhaps you need to install the Gemini dependencies: {exc}. "
            "Try `pip install prompto[gemini]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.vertexai import VertexAIAPI

    ASYNC_APIS["vertexai"] = VertexAIAPI
except ImportError as exc:
    warnings.warn(
        message=(
            f"Vertex AI API ('vertexai') not available. Perhaps you need to install the Vertex AI dependencies: {exc}. "
            "Try `pip install prompto[vertexai]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.ollama import OllamaAPI

    ASYNC_APIS["ollama"] = OllamaAPI
except ImportError as exc:
    warnings.warn(
        message=(
            f"Ollama API ('ollama') not available. Perhaps you need to install the Ollama dependencies: {exc}. "
            "Try `pip install prompto[ollama]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.huggingface_tgi import HuggingfaceTGIAPI

    ASYNC_APIS["huggingface-tgi"] = HuggingfaceTGIAPI
except ImportError as exc:
    warnings.warn(
        message=(
            f"Huggingface TGI API ('huggingface-tgi') not available. Perhaps you need to install the Huggingface TGI dependencies: {exc}. "
            "Try `pip install prompto[huggingface_tgi]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.quart import QuartAPI

    ASYNC_APIS["quart"] = QuartAPI
except ImportError as exc:
    warnings.warn(
        message=(
            f"Quart API ('quart') not available. Perhaps you need to install the Quart dependencies: {exc}. "
            "Try `pip install prompto[quart]`"
        ),
        category=DependencyWarning,
    )

__all__ = ["ASYNC_APIS"]
