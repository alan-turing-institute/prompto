import json

import aiohttp


def get_model_name_identifier(model_name: str) -> str:
    """
    Helper function to get the model name identifier.

    Huggingface model names can contain characters that are not allowed in
    environment variable names. This function replaces those characters
    ("-", "/", ".") with underscores ("_").

    Parameters
    ----------
    model_name : str
        The model name

    Returns
    -------
    str
        The model name identifier with invalid characters replaced
        with underscores
    """
    model_name = model_name.replace("-", "_")
    model_name = model_name.replace("/", "_")
    model_name = model_name.replace(".", "_")

    return model_name


async def async_client_generate(data: dict, url: str, headers: dict) -> dict:
    """
    Asynchronous function to send a POST request to the server.

    Parameters
    ----------
    data : dict
        The data to send in the POST request
    url : str
        The URL to send the POST request to
    headers : dict
        The headers to send with the POST request

    Returns
    -------
    dict
        The JSON response from the server
    """
    # create an asynchronous HTTP session
    async with aiohttp.ClientSession() as session:
        # send the POST request with the data
        async with session.post(
            f"{url}/generate", data=json.dumps(data), headers=headers
        ) as response:
            # check if the response status is OK
            if response.status == 200:
                # return the JSON response
                return await response.json()
            else:
                # return an error message if something went wrong
                raise ValueError(f"Server returned status code {response.status}")
