import sounddevice as sd
import soundfile as sf
import numpy as np
from jetson_nano_asr.common import RecordingConfig, ResampleConfig
from loggy import logger
from torchcodec.decoders import AudioDecoder

from queue import Queue


# Implement a queue to hold audio chunks and a callback function to read audio data from the stream and put it into the queue.
class AudioStream:
    def __init__(self, config: RecordingConfig):
        self.queue = Queue(maxsize=config.max_queue_size)
        self.stream = None
        self.config = config
        self.counter = 0  # Chunks seen so far
        self.use_file = config.file_path is not None
        self.min_required_chunk_size = int(config.sample_rate / 31.25)

        if self.use_file:
            self.file_metadata = (
                AudioDecoder(config.file_path).metadata if self.use_file else None
            )
            self.resample_config = ResampleConfig(
                original_sr=self.file_metadata.sample_rate,
                target_sr=self.config.sample_rate,
            )
        else:
            self.resample_config = ResampleConfig(
                original_sr=sd.query_devices(self.config.device, "input")[
                    "default_samplerate"
                ],
                target_sr=self.config.sample_rate,
            )

        # Silero needs exactly `min_required_chunk_size` samples at target_sr
        # (512 @ 16k, 256 @ 8k). Read enough at the SOURCE rate so that after
        # resampling we land on exactly that — derived, not hardcoded, so mic
        # (48k -> 1536) and 16k file (-> 512) both work.
        self.source_blocksize = round(
            self.min_required_chunk_size
            * self.resample_config.original_sr
            / self.resample_config.target_sr
        )

    def audio_callback(self, indata, frames, time, status) -> None:
        if status:
            logger.debug(
                "Captured audio chunk with {frames} frames at {t:.2f}s",
                frames=frames,
                t=time.inputBufferAdcTime,
            )

        self.queue.put(indata.copy())

    def start_stream(self) -> None:
        if self.use_file:
            with sf.SoundFile(self.config.file_path) as file_stream:
                logger.info("Streaming audio from file: {fp}", fp=self.config.file_path)
                while True:
                    data = file_stream.read(self.source_blocksize, dtype="float32")
                    if not data.size:
                        break
                    self.queue.put(data)
                    self.counter += 1

                self.queue.put(None)

        else:
            self.stream = sd.InputStream(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.resample_config.original_sr,
                blocksize=self.source_blocksize,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()

    def stop_stream(self) -> None:
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def read(self) -> np.ndarray[float] | None:

        if (_chunk := self.queue.get()) is None:
            self.queue.put(None)  # Put None back in the queue for other consumers
            return self.stop_stream()

        self.counter += 1

        resampled = self.resample_config.resample_audio(_chunk.reshape(-1))

        # Pad the final short chunk (the file tail) up to the exact window Silero
        # requires. Padding AFTER resampling, since that's the length the model
        # actually sees. Dropping the tail instead would be fine too (<32ms).
        if resampled.shape[0] < self.min_required_chunk_size:
            resampled = np.pad(
                resampled,
                (0, self.min_required_chunk_size - resampled.shape[0]),
                mode="constant",
                constant_values=0,
            )
        return resampled

    def __enter__(self) -> "AudioStream":
        self.start_stream()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        logger.info(
            "Exiting audio stream context manager | Total chunks read: {chunks}",
            chunks=self.counter,
        )
        self.stop_stream()

        return None
