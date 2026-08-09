import sounddevice as sd
import soundfile as sf
import numpy as np
from jetson_nano_asr.common import RecordingConfig, ResampleConfig
from loggy import logger
from pathlib import Path
import threading


from queue import Queue


# Implement a queue to hold audio chunks and a callback function to read audio data from the stream and put it into the queue.
class AudioStream:
    def __init__(self, config: RecordingConfig):
        self.stream = None
        self.config = config
        self.queue = Queue(maxsize=self.config.max_queue_size)
        self.second_count = 0
        self.counter = 0  # Chunks seen so far
        self.min_required_chunk_size = int(config.sample_rate / 31.25)

        if not self.config.mic:
            self.file_metadata = sf.info(config.file_path)
            self.resample_config = ResampleConfig(
                original_sr=self.file_metadata.samplerate,
                target_sr=self.config.sample_rate,
            )
        else:
            self.resample_config = ResampleConfig(
                original_sr=sd.query_devices(self.config.device, "input")[
                    "default_samplerate"
                ],
                target_sr=self.config.sample_rate,
            )
        self.file_thread: threading.Thread | None = None
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

    def threadable_filestream(self, filepath: Path, chunk_size: int) -> None:
        with sf.SoundFile(filepath) as file_stream:
            for block in file_stream.blocks(
                blocksize=chunk_size, dtype="float32", always_2d=True
            ):
                self.queue.put(block)

        self.queue.put(None)  # Signal the end of the stream
        logger.info("Finished streaming audio from file: {fp}", fp=filepath)

    def start_stream(self) -> None:

        if not self.config.mic:
            logger.info(
                "Starting producer Thread to stream audio from file: {fp}",
                fp=self.config.file_path,
            )
            self.file_thread = threading.Thread(
                target=self.threadable_filestream,
                args=(self.config.file_path, self.source_blocksize),
                daemon=True,
            )
            self.file_thread.start()

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

        if self.file_thread and self.file_thread.is_alive():
            self.file_thread.join(timeout=5)  # Wait for the thread to finish

        if self.stream:

            self.stream.stop()
            self.stream.close()
            self.stream = None

    def read(self) -> np.ndarray[float] | None:

        if (_chunk := self.queue.get()) is None:
            self.queue.put(None)  # Put None back in the queue for other consumers
            return self.stop_stream()
        self.counter += 1
        self.second_count += 1

        if _chunk.shape[1] > 1:
            # logger.warning(
            #     "Audio chunk has {channels} channels, Downmixing channels to mono for VAD processing.",
            #     channels=_chunk.shape[1],
            # )
            _chunk = np.mean(_chunk, axis=1)
            resampled = self.resample_config.resample_audio(_chunk)

        else:
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
        if self.second_count == 32:  # 1 second has passed
            logger.info("1 SECOND HAS PASSED!")
            self.second_count = 0

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
