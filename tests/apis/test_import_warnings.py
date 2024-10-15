import builtins

import pytest

realimport = builtins.__import__

import sys

import regex as re


def test_azure_openai_import_error(monkeypatch):
    def mock_import(name, *args):
        if name == "prompto.apis.azure_openai" and args[2] == ("AzureOpenAIAPI",):
            raise ImportError()
        return realimport(name, *args)

    with monkeypatch.context() as m:
        m.delitem(sys.modules, "prompto.apis")
        m.setattr(builtins, "__import__", mock_import)
        with pytest.warns(
            Warning,
            match=re.escape(
                "Azure OpenAI model ('azure-openai') not available. "
                "Perhaps you need to install the Azure OpenAI dependencies: "
                "Try `pip install prompto[azure_openai]`"
            ),
        ):
            from prompto.apis import ASYNC_APIS


def test_openai_import_error(monkeypatch):
    def mock_import(name, *args):
        if name == "prompto.apis.openai" and args[2] == ("OpenAIAPI",):
            raise ImportError()
        return realimport(name, *args)

    with monkeypatch.context() as m:
        m.delitem(sys.modules, "prompto.apis")
        m.setattr(builtins, "__import__", mock_import)
        with pytest.warns(
            Warning,
            match=re.escape(
                "OpenAI model ('openai') not available. "
                "Perhaps you need to install the OpenAI dependencies: "
                "Try `pip install prompto[openai]`"
            ),
        ):
            from prompto.apis import ASYNC_APIS


def test_anthropic_import_error(monkeypatch):
    def mock_import(name, *args):
        if name == "prompto.apis.anthropic" and args[2] == ("AnthropicAPI",):
            raise ImportError()
        return realimport(name, *args)

    with monkeypatch.context() as m:
        m.delitem(sys.modules, "prompto.apis")
        m.setattr(builtins, "__import__", mock_import)
        with pytest.warns(
            Warning,
            match=re.escape(
                "Anthropic API ('anthropic') not available. "
                "Perhaps you need to install the Anthropic dependencies: "
                "Try `pip install prompto[anthropic]`"
            ),
        ):
            from prompto.apis import ASYNC_APIS


def test_gemini_import_error(monkeypatch):
    def mock_import(name, *args):
        if name == "prompto.apis.gemini" and args[2] == ("GeminiAPI",):
            raise ImportError()
        return realimport(name, *args)

    with monkeypatch.context() as m:
        m.delitem(sys.modules, "prompto.apis")
        m.setattr(builtins, "__import__", mock_import)
        with pytest.warns(
            Warning,
            match=re.escape(
                "Gemini API ('gemini') not available. "
                "Perhaps you need to install the Gemini dependencies: "
                "Try `pip install prompto[gemini]`"
            ),
        ):
            from prompto.apis import ASYNC_APIS


def test_vertexai_import_error(monkeypatch):
    def mock_import(name, *args):
        if name == "prompto.apis.vertexai" and args[2] == ("VertexAIAPI",):
            raise ImportError()
        return realimport(name, *args)

    with monkeypatch.context() as m:
        m.delitem(sys.modules, "prompto.apis")
        m.setattr(builtins, "__import__", mock_import)
        with pytest.warns(
            Warning,
            match=re.escape(
                "Vertex AI API ('vertexai') not available. "
                "Perhaps you need to install the Vertex AI dependencies: "
                "Try `pip install prompto[vertexai]`"
            ),
        ):
            from prompto.apis import ASYNC_APIS


def test_ollama_import_error(monkeypatch):
    def mock_import(name, *args):
        if name == "prompto.apis.ollama" and args[2] == ("OllamaAPI",):
            raise ImportError()
        return realimport(name, *args)

    with monkeypatch.context() as m:
        m.delitem(sys.modules, "prompto.apis")
        m.setattr(builtins, "__import__", mock_import)
        with pytest.warns(
            Warning,
            match=re.escape(
                "Ollama API ('ollama') not available. "
                "Perhaps you need to install the Ollama dependencies: "
                "Try `pip install prompto[ollama]`"
            ),
        ):
            from prompto.apis import ASYNC_APIS


def test_huggingface_tgi_import_error(monkeypatch):
    def mock_import(name, *args):
        if name == "prompto.apis.huggingface_tgi" and args[2] == ("HuggingfaceTGIAPI",):
            raise ImportError()
        return realimport(name, *args)

    with monkeypatch.context() as m:
        m.delitem(sys.modules, "prompto.apis")
        m.setattr(builtins, "__import__", mock_import)
        with pytest.warns(
            Warning,
            match=re.escape(
                "Huggingface TGI API ('huggingface-tgi') not available. "
                "Perhaps you need to install the Huggingface TGI dependencies: "
                "Try `pip install prompto[huggingface_tgi]`"
            ),
        ):
            from prompto.apis import ASYNC_APIS


def test_quart_import_error(monkeypatch):
    def mock_import(name, *args):
        if name == "prompto.apis.quart" and args[2] == ("QuartAPI",):
            raise ImportError()
        return realimport(name, *args)

    with monkeypatch.context() as m:
        m.delitem(sys.modules, "prompto.apis")
        m.setattr(builtins, "__import__", mock_import)
        with pytest.warns(
            Warning,
            match=re.escape(
                "Quart API ('quart') not available. "
                "Perhaps you need to install the Quart dependencies: "
                "Try `pip install prompto[quart]`"
            ),
        ):
            from prompto.apis import ASYNC_APIS
