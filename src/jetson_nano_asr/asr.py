from faster_whisper import WhisperModel

from jetson_nano_asr.vad import VadChunkStream
from jetson_nano_asr.common import RecordingConfig
from loggy import logger
from pathlib import Path

model_size = "base"


def transcribe_audio(model, _chunk) -> str:
    segments, info = model.transcribe(
        _chunk,
        beam_size=1,
        language="en",
        vad_filter=False,
    )

    message = ""
    for segment in segments:

        logger.info(
            "Detected speech from {start:.2f}s to {end:.2f}s: {text}",
            start=segment.start,
            end=segment.end,
            text=segment.text,
        )
        message += segment.text + " "
    return message.strip()


def main(mic_mode: bool, file_path: Path | None) -> None:
    model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

    config = RecordingConfig(mic=mic_mode, file_path=None if mic_mode else file_path)
    vad_stream = VadChunkStream(recording_config=config)

    complete_message = ""
    vad_stream.start_vad()

    while (chunk := vad_stream.read_utterance()) is not None:

        logger.info("Processing chunk of shape: {shape}", shape=chunk.shape)
        complete_message += transcribe_audio(model, chunk)

    # for chunk in vad_stream:
    #     if chunk is None:
    #         logger.info("No more audio chunks available. Exiting.")
    #         break

    #     logger.info("Processing chunk of shape: {shape}", shape=chunk.shape)
    #     complete_message += transcribe_audio(model, chunk)

    logger.info("Final Transcribed Message: {message}", message=complete_message)
