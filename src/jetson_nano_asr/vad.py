import time
import threading
import torch
import numpy as np
from silero_vad import load_silero_vad
from jetson_nano_asr.common import VadConfig, RecordingConfig
from loggy import logger

from jetson_nano_asr.stream import AudioStream
from collections import deque
import queue


torch.set_num_threads(1)


class VadChunkStream:
    """
    Vad Iterator class going over stream.py audio chunks and returning a list of booleans indicating speech presence.

    Whisper processes all audio in 30 seconds = 937.5 chunks of 32ms each.
    We keep a pre-roll buffer of atleast 32 chunks (~1 seconds) to ensure we don't miss the start of speech.

    """

    def __init__(
        self,
        recording_config: RecordingConfig,
        onnx: bool = False,
        config: VadConfig = VadConfig(),
    ):

        self.recording_config = recording_config
        self.model = load_silero_vad(onnx=onnx)
        self.config = config
        self.chunk_size = 32  # 32ms chunks

        self.pre_roll_q = deque(
            maxlen=self.config.preroll_ring_size
        )  # Store the last 10 chunks for pre-roll

        self.consecutive_speech_count = 0  # Count of consecutive chunks with speech
        self.consecutive_silence_count = 0  # Count of consecutive chunks with silence - only meaningful after in_utterance; resets to 0 when speech is detected,+1 on silence; flush firest at N_endpoint  = speech_comp0lete_timeout_ms / chunk_duration_ms

        self.held_chunk = None
        self.last_chunk_timestamp = (
            time.perf_counter()
        )  # Initialize the last chunk timestamp
        self.verbose: bool = True

        self.in_utterance: bool = (
            False  # Flag to indicate if we are currently in an utterance
        )
        self.utterance_buffer: list = (
            []
        )  # Buffer to hold chunks for the current utterance
        self.utterance_queue: queue.Queue = queue.Queue(
            maxsize=recording_config.max_queue_size
        )  # Queue to hold completed utterances

    # def __iter__(self):
    def start_vad(
        self,
    ) -> None:
        """

        Iterates over live mic stream and yields chunks with speech.
        Think about this Condition (Simulating how people talk):
        if a chunk arrives a threshold time after the last detected speech chunk, skip the last chunk and continue to the next one (not implemented yet but gotta think about it).

        a syllable like go, high is ~100ms long, for an average word, we need ~300-500ms of speech.
        """
        self.vad_thread = threading.Thread(target=self._stream_threadable, daemon=True)
        self.vad_thread.start()

    def read_utterance(self) -> np.ndarray | None:
        """
        Reads a completed utterance from the queue.
        Returns None if no utterance is available.
        """

        while True:
            try:
                if _chunk := self.utterance_queue.get_nowait():
                    return _chunk

            except queue.Empty:
                time.sleep(
                    5
                )  # Comment this line later when we want to keep persistent mic stream on

            except Exception as e:
                logger.info("No utterance available in the queue: {error}", error=e)
                return None

    def _stream_threadable(self):
        """
        Threadable function to process audio chunks from the AudioStream and detect speech using VAD.
        Adds chunks to the utterance buffer when speech is detected and yields the buffer when speech ends or max utterance duration is reached.
        """
        with AudioStream(self.recording_config) as audio_stream, torch.no_grad():
            while True:
                audio_chunk = audio_stream.read()
                if audio_chunk is None:
                    # Flush any remaining utterance buffer before exiting
                    if self.utterance_buffer:
                        self.utterance_queue.put(None)

                        self.utterance_buffer.clear()  # Clear the utterance buffer after yielding

                    break

                self.pre_roll_q.append(
                    audio_chunk
                )  # Store the chunk in the pre-roll queue
                logger.info(
                    "Chunk Shape: {shape} | Total chunks: {qsize}",
                    shape=audio_chunk.shape,
                    qsize=audio_stream.queue.qsize(),
                )
                vad_probs = self.model(
                    torch.from_numpy(audio_chunk),
                    sr=audio_stream.config.sample_rate.value,
                ).item()

                # if speech detected
                if vad_probs > 0.5:

                    self.consecutive_speech_count += 1
                    self.consecutive_silence_count = 0

                    self.last_chunk_timestamp = (
                        time.perf_counter()
                    )  # Update the last chunk timestamp
                    self.noise_flag = True

                    if (
                        self.consecutive_speech_count
                        >= self.config.onset_confirm_chunks
                        and not self.in_utterance
                    ):
                        self.in_utterance = True
                        if self.verbose:
                            logger.info(
                                "Speech utterance onset confirmed after {count} consecutive chunks.",
                                count=self.consecutive_speech_count,
                            )
                        # add pre-roll buffer to utterance buffer
                        self.utterance_buffer.extend(self.pre_roll_q)

                    elif self.in_utterance:
                        self.utterance_buffer.append(audio_chunk)
                        if self.verbose:
                            logger.info(
                                "In utterance, appending chunk to utterance buffer. Buffer size: {size}",
                                size=len(self.utterance_buffer),
                            )

                    if self.verbose:
                        logger.info(
                            "Speech detected with probability: {vad_probs:.2f}",
                            vad_probs=vad_probs,
                        )

                else:
                    self.consecutive_silence_count += 1
                    self.consecutive_speech_count = 0

                if (
                    self.consecutive_silence_count
                    >= self.config.speech_complete_timeout_ms // self.chunk_size
                    and self.in_utterance
                ):
                    if self.verbose:
                        logger.info(
                            "Will flush utterance buffer here, as {count} consecutive silence chunks detected. Total buffer duration: {duration:.2f}s",
                            count=self.consecutive_silence_count,
                            duration=len(self.utterance_buffer)
                            * self.chunk_size
                            / 1000,
                        )
                    self.in_utterance = False
                    self.consecutive_silence_count = 0
                    self.consecutive_speech_count = 0

                    self.utterance_queue.put(
                        np.concatenate(self.utterance_buffer, axis=0)
                    )
                    self.utterance_buffer.clear()  # Clear the utterance buffer after yielding

                if self.max_utterance_seconds_reached():
                    if self.verbose:
                        logger.info(
                            "Max utterance duration reached. Flushing buffer. Total buffer duration: {duration:.2f}s",
                            duration=len(self.utterance_buffer)
                            * self.chunk_size
                            / 1000,
                        )

                    self.utterance_queue.put(
                        np.concatenate(self.utterance_buffer, axis=0)
                    )

                    self.utterance_buffer.clear()  # Clear the utterance buffer after yielding
                    self.utterance_buffer.extend(
                        self.pre_roll_q
                    )  # Add pre-roll buffer to the new utterance buffer

    def max_utterance_seconds_reached(self) -> bool:
        """
        Check if the maximum utterance duration has been reached.
        """
        return (
            len(self.utterance_buffer) * self.chunk_size / 1000
            >= self.config.max_utterance_seconds
        )
