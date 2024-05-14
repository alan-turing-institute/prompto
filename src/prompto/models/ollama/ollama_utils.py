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
        return response["response"]
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")
