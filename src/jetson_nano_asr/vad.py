import time

import torch
from silero_vad import load_silero_vad
from .common import RecordingConfig
from loggy import logger

from .stream import AudioStream

torch.set_num_threads(1)


class VadChunkStream:
    """
    Vad Iterator class going over stream.py audio chunks and returning a list of booleans indicating speech presence.
    """

    def __init__(self, onnx: bool = False):
        self.model = load_silero_vad(onnx=onnx)
        self.config = RecordingConfig()
        self.held_chunk = None
        self.noise_flag = False
        self.last_chunk_timestamp = (
            time.perf_counter()
        )  # Initialize the last chunk timestamp

    def __iter__(self):
        """

        Iterates over live mic stream and yields chunks with speech.
        Think about this Condition (Simulating how people talk):
        if a chunk arrives a threshold time after the last detected speech chunk, skip the last chunk and continue to the next one (not implemented yet but gotta think about it).

        """
        with AudioStream(self.config) as audio_stream, torch.no_grad():
            while True:
                audio_chunk = audio_stream.read()
                if audio_chunk is None:
                    # this will be the thread stopping condition, when the stream is closed and no more audio chunks are available
                    yield None  # Yield None to indicate the end of the stream
                    break

                logger.info(
                    "Chunk Shape: {shape} | Total chunks: {qsize}",
                    shape=audio_chunk.shape,
                    qsize=audio_stream.queue.qsize(),
                )
                vad_probs = self.model(
                    torch.from_numpy(audio_chunk),
                    sr=audio_stream.config.sample_rate.value,
                ).item()

                if vad_probs > 0.5:
                    self.last_chunk_timestamp = (
                        time.perf_counter()
                    )  # Update the last chunk timestamp
                    self.noise_flag = True
                    logger.info(
                        "Speech detected with probability: {vad_probs:.2f}",
                        vad_probs=vad_probs,
                    )
                    yield audio_chunk
                else:
                    self.noise_flag = False

    def run(self):
        """
        Run the VAD stream and print the results.
        """
        try:
            for _chunk in self:
                pass

        except KeyboardInterrupt:
            logger.info("VAD stream interrupted by user.")
