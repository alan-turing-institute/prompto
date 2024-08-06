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
except ImportError:
    warnings.warn(
        message=(
            "Azure OpenAI model ('azure-openai') not available. "
            "Perhaps you need to install the Azure OpenAI dependencies: "
            "Try `pip install prompto[azure_openai]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.openai import OpenAIAPI

    ASYNC_APIS["openai"] = OpenAIAPI
except ImportError:
    warnings.warn(
        message=(
            "OpenAI model ('openai') not available. "
            "Perhaps you need to install the OpenAI dependencies: "
            "Try `pip install prompto[openai]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.anthropic import AnthropicAPI

    ASYNC_APIS["anthropic"] = AnthropicAPI
except ImportError:
    warnings.warn(
        message=(
            "Anthropic API ('anthropic') not available. "
            "Perhaps you need to install the Anthropic dependencies: "
            "Try `pip install prompto[anthropic]`"
        ),
        category=DependencyWarning,
    )


try:
    from prompto.apis.gemini import GeminiAPI

    ASYNC_APIS["gemini"] = GeminiAPI
except ImportError:
    warnings.warn(
        message=(
            "Gemini API ('gemini') not available. "
            "Perhaps you need to install the Gemini dependencies: "
            "Try `pip install prompto[gemini]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.vertexai import VertexAIAPI

    ASYNC_APIS["vertexai"] = VertexAIAPI
except ImportError:
    warnings.warn(
        message=(
            "Vertex AI API ('vertexai') not available. "
            "Perhaps you need to install the Vertex AI dependencies: "
            "Try `pip install prompto[vertexai]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.ollama import OllamaAPI

    ASYNC_APIS["ollama"] = OllamaAPI
except ImportError:
    warnings.warn(
        message=(
            "Ollama API ('ollama') not available. "
            "Perhaps you need to install the Ollama dependencies: "
            "Try `pip install prompto[ollama]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.huggingface_tgi import HuggingfaceTGIAPI

    ASYNC_APIS["huggingface-tgi"] = HuggingfaceTGIAPI
except ImportError:
    warnings.warn(
        message=(
            "Huggingface TGI API ('huggingface-tgi') not available. "
            "Perhaps you need to install the Huggingface TGI dependencies: "
            "Try `pip install prompto[huggingface_tgi]`"
        ),
        category=DependencyWarning,
    )

try:
    from prompto.apis.quart import QuartAPI

    ASYNC_APIS["quart"] = QuartAPI
except ImportError:
    warnings.warn(
        message=(
            "Quart API ('quart') not available. "
            "Perhaps you need to install the Quart dependencies: "
            "Try `pip install prompto[quart]`"
        ),
        category=DependencyWarning,
    )

__all__ = ["ASYNC_APIS"]
