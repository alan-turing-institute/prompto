ollama_chat_roles = set(["system", "user", "assistant"])

from ollama import ChatResponse, GenerateResponse


def process_response(response: dict) -> str:
    """
    Helper function to process the response from Ollama API.

    Parameters
    ----------
    response : dict
        The response from the Ollama API as a dictionary

    Returns
    -------
    str
        The processed response text as a string
    """
    if isinstance(response, dict):
        if "response" in response.keys():
            return response["response"]
        elif "message" in response.keys():
            return response["message"]["content"]
        else:
            raise ValueError(
                "Unsupported response format. "
                f"No 'response' or 'message' key found in response: {response}"
            )
    elif isinstance(response, ChatResponse):
        return response.message.content if response.message else ""
    elif isinstance(response, GenerateResponse):
        return response.response if response.response else ""
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")
