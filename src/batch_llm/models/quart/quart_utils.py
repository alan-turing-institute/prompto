import json

import aiohttp


async def async_client_generate(data: dict, url: str, headers: dict) -> dict:

    # Create an asynchronous HTTP session
    async with aiohttp.ClientSession() as session:
        # Send the POST request with the data
        async with session.post(
            url, data=json.dumps(data), headers=headers
        ) as response:
            # Check if the response status is OK
            if response.status == 200:
                # Return the JSON response
                return await response.json()
            else:
                # Return an error message if something went wrong
                raise ValueError(f"Server returned status code {response.status}")
