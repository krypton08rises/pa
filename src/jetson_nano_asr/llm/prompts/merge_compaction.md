You are merging several earlier summary blocks from a voice assistant's conversation history into one, because the number of blocks has grown too large to keep separately. You will be given the summary blocks in chronological order, oldest first, each as a JSON object with the same "facts"/"narrative" schema described below. Produce a single merged block in that same schema.

Merging facts:
- Two facts are the same fact if they describe the same thing, even if worded differently — keep only one.
- If a later fact contradicts or supersedes an earlier one (a preference that changed, a correction to an earlier correction), keep only the later, current fact. Do not mention that something changed or used to be different.
- Keep the original category of each fact you keep.
- Facts that were true only briefly and are no longer relevant given later facts may be dropped entirely.

Merging narrative:
- If an open thread or unresolved question from an earlier block was resolved in a later block (answered, dropped by the user, overtaken by events), remove it — it no longer needs to be carried forward.
- Combine what's left into dense prose. Do not preserve the boundaries between the original blocks or refer to "the earlier summary" / "the later summary" — write it as a single continuous memory.
- If nothing is left to carry forward, output an empty string.

This merged block replaces all of the input blocks — nothing outside of what you output here will be kept from them, so do not drop anything that a later turn of the conversation would still need.

OUTPUT STRUCTURE:
Output a single JSON object with exactly two keys, "facts" and "narrative":

{
  "facts": [
    {"category": "identity" | "environment" | "preference" | "correction" | "instruction", "fact": "<one dense sentence, third person>"}
  ],
  "narrative": "<dense prose covering unresolved questions, open threads, and anything the assistant said it would follow up on that is still outstanding. Empty string if nothing qualifies.>"
}

Output raw JSON only. No markdown code fences, no preamble, no trailing commentary.
