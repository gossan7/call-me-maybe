from typing import List
from pydantic import BaseModel

from llm_sdk import Small_LLM_Model


class Tokenizer(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    llm: Small_LLM_Model

    def encode(self, text: str) -> List[int]:
        return self.llm.encode(text).tolist()[0]

    def decode(self, tokens: List[int]) -> str:
        return self.llm.decode(tokens)
