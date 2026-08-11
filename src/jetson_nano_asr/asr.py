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

    try:
        while (chunk := vad_stream.read_utterance()) is not None:

            # logger.info("Processing chunk of shape: {shape}", shape=chunk.shape)
            msg = transcribe_audio(model, chunk)
            logger.info("Segment Transcribed Message: {message}", message=msg)

            complete_message += msg + "\t"
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping transcription.")
        vad_stream.stop_event.set()  # Signal the VAD thread to stop
        vad_stream.vad_thread.join(timeout=5)
        chunk = None

    except Exception as e:
        logger.error("An error occurred during transcription: {error}", error=e)
        vad_stream.stop_event.set()  # Signal the VAD thread to stop
        chunk = None

    if chunk is not None:
        logger.info("Processing final chunk of shape: {shape}", shape=chunk.shape)
        complete_message += transcribe_audio(model, chunk)

    logger.info("Final Transcribed Message: {message}", message=complete_message)
