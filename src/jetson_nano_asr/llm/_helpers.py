from collections.abc import Callable

from loggy import logger
from pydantic import ValidationError

from ._message import Conversation, SummaryBlock, Message, Content
from ._common import (
    CompactionConfig,
    COMPACTION_PROMPT,
    GEMMA_SYSTEM_PROMPT,
    MERGE_COMPACTION_BLOCKS_PROMPT,
)


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
        Assemble the messages actually sent to the model: one system message
        (the system prompt followed by every summary block, oldest first),
        then the live turns accumulated since the last compaction.

        Summaries are folded into the system message rather than sent as
        separate user turns, so the conversation still starts with a single
        system message and alternates user/assistant after it (Gemma's chat
        template expects that). The prefix only changes at a compaction
        event, so prompt-cache reuse between compactions is unaffected.
        """
        messages = list(self.live.messages)
        has_system = bool(messages) and messages[0].role == "system"
        system_text = messages[0].content if has_system else ""
        turns = messages[1:] if has_system else messages

        summaries = [
            f"[compacted context, {block.source_turn_count} turns]\n{block.content.render()}"
            for block in self.summary_blocks
        ]
        if summaries:
            system_text = "\n\n".join([system_text, *summaries]).strip()

        system = [Message(role="system", content=system_text)] if system_text else []
        return [m.to_openai() for m in system + turns]


def _render_transcript(messages: list[Message]) -> str:
    """Flatten dialogue turns into one plain-text block for the summarizer."""
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def _parse_summary(response: str) -> Content:
    """
    Parse the summarizer's JSON output. Grammar-constrained decoding should
    make this always succeed; the fence-stripping and fallback are here so a
    bad summary degrades into a narrative-only block instead of crashing the
    main loop.
    """
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    try:
        return Content.model_validate_json(text)
    except ValidationError as e:
        logger.warning(
            "Compaction output was not valid JSON, keeping it as narrative: {err}",
            err=e,
        )
        return Content(narrative=text)


class Compactor:
    @classmethod
    def compact_conversation(cls, _query_llm: Callable, cm: ConversationManager):
        """
        Compact the conversation by summarizing the current live conversation and adding it to the summary_blocks.
        """

        messages = cm.live.messages
        has_system = bool(messages) and messages[0].role == "system"
        system_prompt = messages[0].content if has_system else GEMMA_SYSTEM_PROMPT
        turns = messages[1:] if has_system else list(messages)
        if not turns:
            return

        logger.info(
            "Compaction starting: summarizing {turns} messages", turns=len(turns)
        )

        # Separate request: the compaction prompt is the only system message
        # and the dialogue to summarize arrives as a single user message.
        conversation = ConversationManager(live=Conversation())
        conversation.add_system_message(COMPACTION_PROMPT)
        conversation.add_user_message(_render_transcript(turns))
        cm.turn_count += 1

        response = _query_llm(conversation=conversation, compaction=True)
        cm.summary_blocks.append(
            SummaryBlock(
                content=_parse_summary(response),
                token_count=conversation.token_count,
                source_turn_count=len(turns),
                merge_depth=0,
            )
        )

        logger.info(
            "Compaction done: {turns} messages -> summary block #{n} ({tokens} tokens), live reset",
            turns=len(turns),
            n=len(cm.summary_blocks),
            tokens=conversation.token_count,
        )
        if len(cm.summary_blocks) > cm.config.max_compaction_blocks:
            Compactor.merge_compaction(_query_llm=_query_llm, cm=cm)

        # The summary now lives in summary_blocks (rendered into the system
        # message by to_openai), so live restarts with just the system prompt.
        cm.live.clear()
        cm.add_system_message(system_prompt)
        cm.compact_flag = False
        cm.token_count = (
            0  # stale until the next request_llama call reports a real usage count
        )

    @classmethod
    def merge_compaction(cls, _query_llm: Callable, cm: ConversationManager):
        """
        Merge the First two summary blocks into a new summary block.
        """

        # Merge first 2 summary blocks, and add to the front of the list
        conversation = ConversationManager(live=Conversation())
        conversation.add_system_message(MERGE_COMPACTION_BLOCKS_PROMPT)
        conversation.add_user_message(
            "\n\n".join(
                block.content.model_dump_json() for block in cm.summary_blocks[:2]
            )
        )

        response = _query_llm(conversation=conversation, compaction=True)
        content = _parse_summary(response)
        cm.summary_blocks = [
            SummaryBlock(
                content=content,
                token_count=conversation.token_count,
                source_turn_count=cm.summary_blocks[0].source_turn_count
                + cm.summary_blocks[1].source_turn_count,
                merge_depth=max(
                    cm.summary_blocks[0].merge_depth, cm.summary_blocks[1].merge_depth
                )
                + 1,
            )
        ] + cm.summary_blocks[2:]

        logger.info(
            "Merge compaction done: first two summary blocks merged into new summary block #{n} ({tokens} tokens)",
            n=len(cm.summary_blocks),
            tokens=conversation.token_count,
        )
