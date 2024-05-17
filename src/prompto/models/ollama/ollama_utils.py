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


def sort_ollama_prompts(prompt_dicts: list[dict]) -> list[dict]:
    """
    For a list of prompt dictionaries, sort the dictionaries with "api": "ollama"
    by the "model_name" key. The rest of the dictionaries are kept in the same order.

    For Ollama API, if the model requested is not currently loaded, the model will be
    loaded on demand. This can take some time, so it is better to sort the prompts
    by the model name to reduce the time taken to load the models.

    If no "ollama" dictionaries are present, the original list is returned.

    Parameters
    ----------
    prompt_dicts : list[dict]
        List of dictionaries containing the prompt and other parameters
        to be sent to the API. Each dictionary must have keys "prompt" and "api".

    Returns
    -------
    list[dict]
        List of dictionaries containing the prompt and other parameters
        where the "ollama" dictionaries are sorted by the "model_name" key
    """
    ollama_indices = [
        i for i, item in enumerate(prompt_dicts) if item.get("api") == "ollama"
    ]
    if len(ollama_indices) == 0:
        return prompt_dicts

    # sort indices for "ollama" dictionaries
    sorted_ollama_indices = sorted(
        ollama_indices, key=lambda i: prompt_dicts[i].get("model_name", "")
    )

    # create map from original ollama index to sorted index
    ollama_index_map = {i: j for i, j in zip(ollama_indices, sorted_ollama_indices)}

    # sort data based on the combined indices
    return [
        (
            prompt_dicts[i]
            if i not in ollama_index_map.keys()
            else prompt_dicts[ollama_index_map[i]]
        )
        for i in range(len(prompt_dicts))
    ]
