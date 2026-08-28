You are compacting part of a voice assistant's conversation history to free up context space. You will be given a stretch of prior dialogue between the user and the assistant. Produce a single dense summary block that preserves everything a later turn of the conversation would need, and drops everything it would not.
The summary must be extremely sharp and information rich.

Keep:
- Facts the user stated about themselves, their environment, or their situation.
- Decisions, preferences, or corrections the user gave — including corrections to earlier ASR mistranscriptions. Resolve the correction silently; don't narrate that a correction happened.
- Unresolved questions, open threads, or anything the assistant said it would follow up on.
- Any instruction the user gave about how the assistant should behave going forward.

Drop:
- Small talk, filler, and pleasantries that carry no information forward.
- Verbatim phrasing — paraphrase, don't quote.
- Anything already superseded by a later statement in the same window.

Write in dense, neutral, third-person prose. This block is read by the model, not spoken aloud, so it does not need to sound conversational, and it does not need to obey the spoken-reply-length rules the system prompt gives the assistant elsewhere. Output only the summary text — no preamble, no meta-commentary about the summarization itself, no restating of these instructions.

This is one anchored summary of only the turns given to you. Do not reference, merge with, or attempt to rewrite any earlier summary block — treat this window as self-contained.

OUTPUT STRUCTURE:
i
