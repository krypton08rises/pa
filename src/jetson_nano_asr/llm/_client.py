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
    logger.debug(
        "Sending {kind} request: {n} messages",
        kind="compaction" if compaction else "chat",
        n=len(messages),
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
            logger.debug(
                "LLM usage: {total_tokens} total tokens",
                total_tokens=conversation.token_count,
            )

    conversation.add_assistant_message(contents)
    return contents
