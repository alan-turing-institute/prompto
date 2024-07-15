from anthropic.types.message import Message

def process_response(response: Message) -> str | list[str]:
    """
    Helper function to process the response from the Anthropic API.

    Parameters
    ----------
    response : Message
        The response from the Anthropic API

    Returns
    -------
    str | list[str]
        The processed response. If there are multiple responses,
        a list of strings is returned, otherwise a single string is returned
    """
    assert isinstance(response, Message), f"Unsupported response type: {type(response)}"
    if len(response.content) == 0:
        return ""
    elif len(response.content) == 1:
        return response.content[0].text
    else:
        return [choice.text for choice in response.content]


