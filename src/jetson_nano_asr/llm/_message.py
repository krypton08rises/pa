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


FactCategory = Literal[
    "identity", "environment", "preference", "correction", "instruction"
]


class Fact(BaseModel):
    category: FactCategory
    fact: str


class Content(BaseModel):
    """
    Structured compaction output: atomic, mergeable facts plus a narrative
    residue for whatever doesn't reduce to a clean fact (open threads,
    unresolved questions, anything that needs surrounding context).
    """

    facts: list[Fact] = []
    narrative: str = ""

    def render(self) -> str:
        blocks = []
        if self.facts:
            blocks.append(
                "Facts:\n" + "\n".join(f"- ({f.category}) {f.fact}" for f in self.facts)
            )
        if self.narrative:
            blocks.append(f"Narrative:\n{self.narrative}")
        return "\n\n".join(blocks)


class SummaryBlock(BaseModel):

    content: Content
    token_count: int
    source_turn_count: int  # Number of turns in the conversation that were compacted into this summary block.

    merge_depth: int = (
        0  # How many times this summary block has been merged into a new summary block.
    )
