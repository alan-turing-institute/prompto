import json

import aiohttp


async def async_client_generate(data: dict, url: str, headers: dict) -> dict:
    # create an asynchronous HTTP session
    async with aiohttp.ClientSession() as session:
        # send the POST request with the data
        async with session.post(
            url, data=json.dumps(data), headers=headers
        ) as response:
            # check if the response status is OK
            if response.status == 200:
                # return the JSON response
                return await response.json()
            else:
                # return an error message if something went wrong
                raise ValueError(f"Server returned status code {response.status}")
