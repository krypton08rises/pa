from typing import Literal, Any, List
from pydantic import BaseModel

# Bedrock does not allow 'system' in the message array, so we must handle it separately.
Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: Role
    content: str

    def to_openai(self) -> dict[str, Any]:
        """Returns a valid openai.types.chat.ChatCompletionMessageParam"""
        return {"role": self.role, "content": self.content}


class Conversation:

    def __init__(self, messages: List[Message] | None = None):
        self.messages: List[Message] = messages if messages is not None else []

    def add_message(self, role: Role, content: str):
        self.messages.append(Message(role=role, content=content))

    def clear(self):
        self.messages.clear()


class SummaryBlock(BaseModel):

    content: str
    token_count: int
    source_turn_count: int  # Number of turns in the conversation that were compacted into this summary block.

    merge_depth: int = (
        0  # How many times this summary block has been merged into a new summary block.
    )
