import time
import torch
import queue
import threading

from faster_whisper import WhisperModel

from jetson_nano_asr.vad import VadChunkStream
from jetson_nano_asr.common import RecordingConfig, VadConfig, WHISPER_MODEL_SIZE
from loggy import logger
from pathlib import Path

torch.set_num_threads(1)


class WhisperTranscriber:
    def __init__(
        self,
        mic_mode: bool,
        file_path: Path | None,
        model_size: str = WHISPER_MODEL_SIZE,
    ):

        self.mic_mode = mic_mode
        self.file_path = file_path
        self.model = WhisperModel(
            model_size, device="cuda", compute_type="int8_float16"
        )
        self.config = RecordingConfig(
            mic=self.mic_mode, file_path=None if self.mic_mode else self.file_path
        )

        self.speech_q = queue.Queue(
            self.config.speech_max_queue_size
        )  # Queue to hold transcribed speech segments
        self.stop_event = threading.Event()  # Event to signal the VAD thread to stop

    def transcribe_audio(self, _chunk) -> str:
        segments, info = self.model.transcribe(
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

    def start_transcription(
        self,
    ):

        self.asr_thread = threading.Thread(
            target=self.threadable_asr,
        )
        self.asr_thread.start()

    def read_transcripts(self) -> str:
        """
        Read transcribed messages from the speech queue.
        """
        while True:
            try:
                if (_speech := self.speech_q.get_nowait()) is not None:
                    return _speech
            except queue.Empty:
                if self.stop_event.is_set():
                    logger.info("Transcription stopped. No more messages.")
                    return None
                time.sleep(5)  # Wait for a second before checking again
            except Exception as e:
                logger.info("No speech available in the queue: {error}", error=e)
                return None

    def threadable_asr(self) -> None:
        """
        Main function to transcribe audio from a microphone or a file using the Whisper model.
        """

        vad_config = VadConfig()
        vad_stream = VadChunkStream(
            recording_config=self.config, config=vad_config, onnx=False, verbose=False
        )

        complete_message = ""
        vad_stream.start_vad()

        try:
            while (chunk := vad_stream.read_utterance()) is not None:

                # logger.info("Processing chunk of shape: {shape}", shape=chunk.shape)
                msg = self.transcribe_audio(chunk)
                logger.info("Segment Transcribed Message: {message}", message=msg)

                complete_message += msg + "\t"
                self.speech_q.put(msg)  # Put the transcribed message into the queue

                if self.stop_event.is_set():
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Stopping transcription.")
            vad_stream.stop_event.set()  # Signal the VAD thread to stop
            vad_stream.vad_thread.join(timeout=5)

        except Exception as e:
            logger.error("An error occurred during transcription: {error}", error=e)
            vad_stream.stop_event.set()  # Signal the VAD thread to stop

        self.stop_event.set()  # Signal the ASR thread to stop

        logger.info("Final Transcribed Message: {message}", message=complete_message)
