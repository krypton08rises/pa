from collections.abc import Callable

from loggy import logger

from ._message import Conversation, SummaryBlock, Message
from ._common import CompactionConfig, COMPACTION_PROMPT, GEMMA_SYSTEM_PROMPT


class ConversationManager:

    def __init__(
        self, live: Conversation | None, config: CompactionConfig | None = None
    ):
        self.live: Conversation = live
        self.summary_blocks: list[SummaryBlock] = []
        self.token_count: int = 0
        self.idle_time: float = 0
        self.turn_count: int = 0
        self.config: CompactionConfig = (
            config if config is not None else CompactionConfig()
        )

        self.compact_flag: bool = False

    def add_system_message(self, content: str):
        """
        Add a system message to the conversation.
        """
        self.live.add_message(role="system", content=content)

    def add_user_message(self, content: str):
        """
        Add a user message to the conversation.
        """
        self.live.add_message(role="user", content=content)

    def add_assistant_message(self, content: str):
        """
        Add an assistant message to the conversation.
        """
        self.live.add_message(role="assistant", content=content)

    def add_tool_message(self, content: str):
        """
        Add a tool message to the conversation.
        """
        self.live.add_message(role="tool", content=content)

    def check_compaction(self, _llm_call: Callable) -> bool:
        """
        Check if the conversation is close to the max_length and set compaction_due to True if it is.
        """
        if (
            self.token_count
            > self.config.forced_compaction_token_limit * self.config.max_tokens
        ):
            logger.info(
                "Hard compaction threshold hit: {tokens}/{max} tokens ({pct:.0%}) — compacting now",
                tokens=self.token_count,
                max=self.config.max_tokens,
                pct=self.token_count / self.config.max_tokens,
            )
            self.compact_flag = True
            Compactor.compact_conversation(_query_llm=_llm_call, cm=self)
            return True

        if (
            self.token_count
            > self.config.soft_compaction_token_limit * self.config.max_tokens
            and self.idle_time > self.config.idle_time_limit
        ):
            logger.info(
                "Soft compaction threshold hit while idle: {tokens}/{max} tokens ({pct:.0%}), idle {idle:.1f}s — compacting now",
                tokens=self.token_count,
                max=self.config.max_tokens,
                pct=self.token_count / self.config.max_tokens,
                idle=self.idle_time,
            )
            self.compact_flag = True
            Compactor.compact_conversation(_query_llm=_llm_call, cm=self)
            return True

        return False

    def to_openai(self) -> list[dict]:
        """
        Assemble the messages actually sent to the model: the system message,
        one message per summary block (oldest first), then the live turns
        accumulated since the last compaction.
        """
        messages = list(self.live.messages)
        system = messages[:1] if messages and messages[0].role == "system" else []
        turns = messages[1:] if system else messages

        summary_messages = [
            Message(
                role="user",
                content=f"[compacted context, {block.source_turn_count} turns]\n{block.content}",
            )
            for block in self.summary_blocks
        ]

        return [m.to_openai() for m in system + summary_messages + turns]


class Compactor:
    @classmethod
    def compact_conversation(cls, _query_llm: Callable, cm: ConversationManager):
        """
        Compact the conversation by summarizing the current live conversation and adding it to the summary_blocks.
        """

        turn_count = len(cm.live.messages)
        logger.info(
            "Compaction starting: summarizing {turns} messages", turns=turn_count
        )

        # plan is to add compaction prompt to a completely seperate object
        conversation = ConversationManager(
            live=Conversation(messages=list(cm.live.messages))
        )
        conversation.add_system_message(COMPACTION_PROMPT)
        # This would empty the live
        cm.turn_count += 1

        response = _query_llm(conversation=conversation, compaction=True)
        cm.summary_blocks.append(
            SummaryBlock(
                content=response,
                token_count=conversation.token_count,
                source_turn_count=len(cm.live.messages),
                merge_depth=0,
            )
        )
        cm.live.clear()
        cm.compact_flag = False
        cm.add_system_message(GEMMA_SYSTEM_PROMPT)
        cm.add_assistant_message(response)

        logger.info(
            "Compaction done: {turns} messages -> summary block #{n} ({tokens} tokens), live reset",
            turns=turn_count,
            n=len(cm.summary_blocks),
            tokens=conversation.token_count,
        )

    def merge_compaction(self, _query_llm: Callable, cm: ConversationManager):
        """
        Merge the last two summary blocks into a new summary block.
        """
        pass
