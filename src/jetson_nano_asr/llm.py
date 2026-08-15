from openai import OpenAI

from loggy import logger
from jetson_nano_asr.asr import WhisperTranscriber
from jetson_nano_asr.common import BASE_URL, GEMMA_SYSTEM_PROMPT


def request_llama(conversation: list[dict[str, str]]):
    """
    Make a request to the local llama server.
    """

    client = OpenAI(
        base_url=BASE_URL, api_key="sk-no-key-required"  # The local server ignores this
    )

    response = client.chat.completions.create(
        model="local-gemma-26B",  # Ignored by server but Sdk requires it
        messages=conversation,
        temperature=1,
        stream=True,
    )
    contents = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            contents += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)
    return contents


def main():
    """
    Main function to route whisper transcriptions to the local llama server for processing.
    """
    conversation = [
        {"role": "system", "content": GEMMA_SYSTEM_PROMPT},
    ]

    transcriber = WhisperTranscriber(mic_mode=True, file_path=None)
    transcriber.start_transcription()

    while (transcript := transcriber.read_transcripts()) is not None:

        conversation.append({"role": "user", "content": transcript})

        llm_response = request_llama(conversation=conversation)
        logger.info("LLM Response: {response}", response=llm_response)
        conversation.append({"role": "assistant", "content": llm_response})


if __name__ == "__main__":
    main()
