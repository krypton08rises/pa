import time
from loggy import logger

from jetson_nano_asr.asr import WhisperTranscriber
from jetson_nano_asr.llm._common import GEMMA_SYSTEM_PROMPT
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
        turn_start = time.perf_counter()
        llm_response = request_llama(conversation=conversation)
        turn_elapsed = time.perf_counter() - turn_start
        logger.info(
            "Turn: user {transcript!r} -> assistant {response!r} ({chars} chars, {elapsed:.2f}s)",
            transcript=transcript,
            response=llm_response,
            chars=len(llm_response),
            elapsed=turn_elapsed,
        )

        conversation.idle_time = time.perf_counter() - transcriber.idle_time_last

        _compact = conversation.check_compaction(_llm_call=request_llama)
        logger.info(
            "Turn stats: {token_count}/{max_tokens} tokens ({token_pct:.0%}), idle {idle_time:.2f}s "
            "(limit {idle_limit:.1f}s), {n_blocks} summary blocks, compaction_due={compaction_due}",
            token_count=conversation.token_count,
            max_tokens=conversation.config.max_tokens,
            token_pct=conversation.token_count / conversation.config.max_tokens,
            idle_time=conversation.idle_time,
            idle_limit=conversation.config.idle_time_limit,
            n_blocks=len(conversation.summary_blocks),
            compaction_due=_compact,
        )
