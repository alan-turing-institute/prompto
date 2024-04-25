def process_response(response: dict) -> str:
    if isinstance(response, dict):
        return response["response"]
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")
