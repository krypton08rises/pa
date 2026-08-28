from pathlib import Path


def read_system_prompt(fp: Path) -> str:
    """
    Read the system prompt from the GEMMA_SYSTEM_PROMPT.md file.
    """

    with open(fp, "r") as f:
        return f.read()


GEMMA_SYSTEM_PROMPT = read_system_prompt(Path(__file__).parent / "prompts/system.md")
COMPACTION_PROMPT = read_system_prompt(Path(__file__).parent / "prompts/compaction.md")
MERGE_COMPACTION_BLOCKS = read_system_prompt(
    Path(__file__).parent / "prompts/merge_compaction.md"
)


class CompactionConfig:

    def __init__(
        self,
        max_compaction_blocks: int = 16,
        max_tokens: int = 8192,
        forced_compaction_token_limit: float = 0.8,
        idle_time_limit: float = 10.0,
        soft_compaction_token_limit: float = 0.1,
    ):
        self.max_compaction_blocks: int = max_compaction_blocks
        self.max_tokens: int = max_tokens
        self.forced_compaction_token_limit: float = (
            forced_compaction_token_limit  # Trigger
        )
        self.idle_time_limit: float = (
            idle_time_limit  # Trigger soft compaction when idle time exceeds 10 seconds
        )
        self.soft_compaction_token_limit: float = (
            soft_compaction_token_limit  # Trigger compaction when estimated token count exceeds 60% of max_tokens when idle time > 10 seconds
        )
