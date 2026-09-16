import time

from openai import OpenAI
from loggy import logger
from jetson_nano_asr.common import BASE_URL
from jetson_nano_asr.llm._helpers import ConversationManager


def request_llama(conversation: ConversationManager, compaction: bool = False) -> str:
    """
    Make a request to the local llama server.
    """

    client = OpenAI(
        base_url=BASE_URL, api_key="sk-no-key-required"  # The local server ignores this
    )

    messages = conversation.to_openai()
    request_start = time.perf_counter()
    logger.debug(
        "Sending {kind} request: {n} messages, {chars} total chars, {pre_tokens} tokens known going in",
        kind="compaction" if compaction else "chat",
        n=len(messages),
        chars=sum(len(m["content"]) for m in messages),
        pre_tokens=conversation.token_count,
    )

    response = client.chat.completions.create(
        model="local-gemma-26B",  # Ignored by server but Sdk requires it
        messages=messages,
        temperature=1,
        stream=True,
        stream_options={"include_usage": True},
    )
    contents = ""

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            contents += chunk.choices[0].delta.content

        if chunk.usage:
            conversation.token_count = chunk.usage.total_tokens
            elapsed = time.perf_counter() - request_start
            logger.debug(
                "LLM usage ({kind}): {prompt} prompt + {completion} completion = {total} total tokens, {elapsed:.2f}s elapsed",
                kind="compaction" if compaction else "chat",
                prompt=chunk.usage.prompt_tokens,
                completion=chunk.usage.completion_tokens,
                total=conversation.token_count,
                elapsed=elapsed,
            )

    conversation.add_assistant_message(contents)
    return contents
