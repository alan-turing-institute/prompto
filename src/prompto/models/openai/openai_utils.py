from enum import Enum

from openai.types.chat import ChatCompletion
from openai.types.completion import Completion


class ChatRoles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def process_response(response: ChatCompletion | Completion) -> str | list[str]:
    if isinstance(response, ChatCompletion):
        if len(response.choices) == 0:
            return ""
        elif len(response.choices) == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]
    elif isinstance(response, Completion):
        if len(response.choices) == 0:
            return ""
        elif len(response.choices) == 1:
            return response.choices[0].text
        else:
            return [choice.text for choice in response.choices]
    else:
        raise ValueError(f"Unsupported response type: {type(response)}")
