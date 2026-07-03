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
                    data = file_stream.read(self.config.chunk_size, dtype="float32")
                    if not data.size:
                        break
                    logger.info(
                        "Data size: {ds} | Total chunks: {qsize}",
                        ds=data.size,
                        qsize=self.queue.qsize(),
                    )
                    self.queue.put(data)
                    self.counter += 1

                logger.info(
                    "Finished streaming audio from file | Total chunks: {chunks}",
                    chunks=self.queue.qsize(),
                )
                self.queue.put(None)

        else:
            self.stream = sd.InputStream(
                device=self.config.device,
                channels=self.config.channels,
                samplerate=self.resample_config.original_sr,
                blocksize=self.config.chunk_size,
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

        if _chunk.size < self.min_required_chunk_size:
            _chunk = np.pad(
                _chunk,
                (0, int(self.min_required_chunk_size - _chunk.shape[0])),
                mode="constant",
                constant_values=0,
            )
        return self.resample_config.resample_audio(
            _chunk.reshape(
                -1,
            )
        )

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
