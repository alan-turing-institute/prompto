from enum import Enum

from openai.types.chat import ChatCompletion
from openai.types.completion import Completion


class ChatRoles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def process_response(response: ChatCompletion | Completion) -> str:
    if isinstance(response, ChatCompletion):
        return response.choices[0].message.content
    elif isinstance(response, Completion):
        return response.choices[0].text
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")
