from prompto.apis import (
    AnthropicAPI,
    AzureOpenAIAPI,
    GeminiAPI,
    HuggingfaceTGIAPI,
    OllamaAPI,
    OpenAIAPI,
    QuartAPI,
    VertexAIAPI,
)


def test_azure_openai_import():
    from prompto.apis import ASYNC_APIS

    assert "azure-openai" in ASYNC_APIS
    assert ASYNC_APIS["azure-openai"] == AzureOpenAIAPI


def test_openai_import(monkeypatch):
    from prompto.apis import ASYNC_APIS

    assert "openai" in ASYNC_APIS
    assert ASYNC_APIS["openai"] == OpenAIAPI


def test_anthropic_import():
    from prompto.apis import ASYNC_APIS

    assert "anthropic" in ASYNC_APIS
    assert ASYNC_APIS["anthropic"] == AnthropicAPI


def test_gemini_import():
    from prompto.apis import ASYNC_APIS

    assert "gemini" in ASYNC_APIS
    assert ASYNC_APIS["gemini"] == GeminiAPI


def test_vertexai_import():
    from prompto.apis import ASYNC_APIS

    assert "vertexai" in ASYNC_APIS
    assert ASYNC_APIS["vertexai"] == VertexAIAPI


def test_ollama_import():
    from prompto.apis import ASYNC_APIS

    assert "ollama" in ASYNC_APIS
    assert ASYNC_APIS["ollama"] == OllamaAPI


def test_huggingface_tgi_import():
    from prompto.apis import ASYNC_APIS

    assert "huggingface-tgi" in ASYNC_APIS
    assert ASYNC_APIS["huggingface-tgi"] == HuggingfaceTGIAPI


def test_quart_import():
    from prompto.apis import ASYNC_APIS

    assert "quart" in ASYNC_APIS
    assert ASYNC_APIS["quart"] == QuartAPI
