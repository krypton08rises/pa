from faster_whisper import WhisperModel

from jetson_nano_asr.vad import VadChunkStream
from loggy import logger


model_size = "base"

"""
try with test.wav
"""


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


def main():
    model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    vad_stream = VadChunkStream(onnx=False)

    complete_message = ""
    for chunk in vad_stream:
        if chunk is None:
            logger.info("No more audio chunks available. Exiting.")
            break

        logger.info("Processing chunk of shape: {shape}", shape=chunk.shape)
        complete_message += transcribe_audio(model, chunk)

    logger.info("Final Transcribed Message: {message}", message=complete_message)


if __name__ == "__main__":
    main()
