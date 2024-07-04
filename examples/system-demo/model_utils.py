from enum import Enum, auto


def send_openai_gpt3_5(prompt: str, params: dict[int]):
    print(f"send 3.5: {prompt}")


class Models(Enum):
    OPENAI_GPT3_5 = auto
    OPENAI_GPT4 = auto()
    GEMINI = auto()
    OLLAMA = auto()

    def send_prompt(self, prompt_dict):
        match prompt_dict:
            # with params
            case {"api": api, "prompt": prompt, "parameters": params}:
                match self:
                    case Models.OPENAI_GPT3_5:
                        assert api == "gpt-3.5-turbo"
                        send_openai_gpt3_5(prompt, params)

                    case Models.OPENAI_GPT4:
                        print(f"send 4: {prompt}")

            # without params
            case {"api": api, "prompt": prompt}:
                pass
