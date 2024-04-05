from batch_llm.models.gemini.gemini import Gemini
from batch_llm.models.testing.testing_model import TestModel

MODELS = {
    "test": TestModel,
    "gemini": Gemini,
}

__all__ = ["MODELS"]
