import os

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
from tqdm import tqdm


def send_prompts_sync(prompt_dicts: list[dict]) -> list[str]:
    # maive for loop to synchronously dispatch prompts
    return [send_prompt(prompt_dict) for prompt_dict in tqdm(prompt_dicts)]


def send_prompt(prompt_dict: dict) -> str:
    # function to send a prompt to the appropriate API (OpenAI, Gemini, or Ollama)
    try:
        match prompt_dict:
            # with params
            case {
                "id": _,
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
    except (Exception, BaseException) as err:
        return f"Error: {type(err).__name__} - {err}"


def send_openai(prompt: str, model_name: str, params: dict[int]) -> str:
    # function to send a prompt to the OpenAI API
    # obtain the API key from the environment
    api_key = os.environ.get("OPENAI_API_KEY")

    # set up (synchronous) OpenAI client
    openai.api_key = api_key
    openai.api_type = "openai"
    client = OpenAI(api_key=api_key)

    # send the prompt to the OpenAI API
    response: ChatCompletion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        **params,
    )

    return response.choices[0].message.content


def send_gemini(prompt: str, model_name: str, params: dict[int]) -> str:
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
    response: GenerateContentResponse = GenerativeModel(model_name).generate_content(
        contents=prompt,
        generation_config=params,
        safety_settings=safety_settings,
        stream=False,
    )

    return response.candidates[0].content.parts[0].text


def send_ollama(prompt: str, model_name: str, params: dict[int]) -> str:
    # function to send a prompt to the Ollama API
    # obtain the API endpoint from the environment
    endpoint = os.environ.get("OLLAMA_API_ENDPOINT")

    # set up the Ollama client
    client = Client(host=endpoint)

    # send the prompt to the Ollama API
    response: dict = client.generate(
        model=model_name,
        prompt=prompt,
        options=params,
    )

    return response["response"]
