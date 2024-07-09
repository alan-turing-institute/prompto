import os
from enum import Enum, auto
from typing import Any

import google.generativeai as genai
import openai
from google.generativeai import GenerativeModel
from google.generativeai.types import (
    GenerateContentResponse,
    HarmBlockThreshold,
    HarmCategory,
)
from ollama import Client
from openai import OpenAI
from openai.types.chat import ChatCompletion


def send_prompt(prompt_dict: dict) -> Any:
    # function to send a prompt to the appropriate API (OpenAI, Gemini, or Ollama)
    match prompt_dict:
        # with params
        case {
            "api": api,
            "model_name": model_name,
            "prompt": prompt,
            "parameters": params,
        }:
            match api:
                case "openai":
                    return send_openai(prompt, model_name, params)

                case "gemini":
                    return send_gemini(prompt, model_name, params)

                case "ollama":
                    return send_ollama(prompt, model_name, params)

                case _:
                    raise ValueError(f"Unsupported API: {api}")

        case _:
            raise ValueError("Invalid prompt dictionary")


def send_openai(prompt: str, model_name: str, params: dict[int]) -> ChatCompletion:
    # function to send a prompt to the OpenAI API
    # obtain the API key from the environment
    api_key = os.environ.get("OPENAI_API_KEY")

    # set up (synchronous) OpenAI client
    openai.api_key = api_key
    openai.api_type = "openai"
    client = OpenAI(api_key=api_key)

    # send the prompt to the OpenAI API
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        **params,
    )

    return response


def send_gemini(
    prompt: str, model_name: str, params: dict[int]
) -> GenerateContentResponse:
    # function to send a prompt to the Gemini API
    # obtain the API key from the environment
    api_key = os.environ.get("GEMINI_API_KEY")

    # configure the API key
    genai.configure(api_key=api_key)

    # set up default safety settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    # send the prompt to the Gemini API
    response = GenerativeModel(model_name).generate_content(
        contents=prompt,
        generation_config=params,
        safety_settings=safety_settings,
        stream=False,
    )

    return response


def send_ollama(prompt: str, model_name: str, params: dict[int]) -> dict:
    # function to send a prompt to the Ollama API
    # obtain the API endpoint from the environment
    endpoint = os.environ.get("OLLAMA_API_ENDPOINT")

    # set up the Ollama client
    client = Client(host=endpoint)

    # send the prompt to the Ollama API
    response = client.generate(
        model=model_name,
        prompt=prompt,
        options=params,
    )

    return response


# class API(Enum):
#     OPENAI = auto()
#     GEMINI = auto()
#     OLLAMA = auto()

#     def send_prompt(self, prompt_dict):
#         match prompt_dict:
#             # with params
#             case {"api": api, "model_name": model_name, "prompt": prompt, "parameters": params}:
#                 match self:
#                     case API.OPENAI:
#                         assert api == "openai"
#                         send_openai(prompt, model_name, params)

#                     case API.GEMINI:
#                         assert api == "gemini"
#                         pass

#                     case API.OLLAMA:
#                         assert api == "ollama"
#                         pass
