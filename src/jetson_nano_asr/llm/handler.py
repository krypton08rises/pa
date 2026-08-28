import time
from loggy import logger

from jetson_nano_asr.asr import WhisperTranscriber
from jetson_nano_asr.common import GEMMA_SYSTEM_PROMPT
from jetson_nano_asr.llm._helpers import ConversationManager
from jetson_nano_asr.llm._client import request_llama
from jetson_nano_asr.llm._message import Conversation


def main():
    """
    Main function to route whisper transcriptions to the local llama server for processing.
    """

    transcriber = WhisperTranscriber(mic_mode=True, file_path=None)
    transcriber.start_transcription()

    conversation = ConversationManager(live=Conversation())
    conversation.add_system_message(GEMMA_SYSTEM_PROMPT)

    while (transcript := transcriber.read_transcripts()) is not None:
        conversation.add_user_message(transcript)
        # conversation.append({"role": "user", "content": transcript})
        llm_response = request_llama(conversation=conversation)
        logger.info("LLM Response: {response}", response=llm_response)

        conversation.idle_time = time.perf_counter() - transcriber.idle_time_last

        _compact = conversation.check_compaction(_llm_call=request_llama)
        logger.debug(
            "Conversation token count: {token_count}, idle time: {idle_time:.2f}s, compaction due: {compaction_due}",
            token_count=conversation.token_count,
            idle_time=conversation.idle_time,
            compaction_due=_compact,
        )
        # conversation.append({"role": "assistant", "content": llm_response})
