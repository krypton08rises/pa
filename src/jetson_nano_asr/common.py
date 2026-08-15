import librosa
import numpy as np

from enum import IntEnum
from pathlib import Path
from typing import Literal, NewType

from loggy import logger

BASE_URL = "http://localhost:8080/v1"

MAX_BUFFER_SIZE = 10  # Seconds?

DEVICE = 14

WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large-v1, large-v2


def read_system_prompt(fp: Path) -> str:
    """
    Read the system prompt from the GEMMA_SYSTEM_PROMPT.md file.
    """

    with open(fp, "r") as f:
        return f.read()


GEMMA_SYSTEM_PROMPT = read_system_prompt(
    Path(__file__).parent / "data/llama_prompts/system.md"
)

ResampleType = NewType(
    "ResampleType",
    Literal[
        "kaiser_best",
        "kaiser_fast",
        "fft",
        "polyphase",
        "soxr_qq",
        "soxr_vhq",
        "soxr_hq",
        "soxr_mq",
        "soxr_lq",
    ],
)


class SampleRate(IntEnum):
    EXPECTED: int = 48000
    DEFAULT: int = 16_000
    HALF_SUPPORT: int = 8_000


class Channels(IntEnum):
    MONO: int = 1
    STEREO: int = 2
    SURROUND: int = 6


class RecordingConfig:
    def __init__(
        self,
        audio_device: int = DEVICE,
        mic: bool = True,
        file_path: Path | None = None,
        sample_rate: SampleRate = SampleRate.DEFAULT,
        channels: Channels = Channels.MONO,
        max_queue_size: int = 1024,
        utterance_max_queue_size: int = 10,
        speech_max_queue_size: int = 10,
    ):

        self.mic: bool = mic
        self.device: int = audio_device
        if not mic and file_path is None:
            raise ValueError(
                "Either mic must be True or a valid file_path must be provided."
            )

        self.file_path: Path | None = file_path
        self.sample_rate: SampleRate = sample_rate
        self.channels: Channels = channels
        self.max_queue_size: int = max_queue_size
        self.utterance_max_queue_size: int = utterance_max_queue_size
        self.speech_max_queue_size: int = speech_max_queue_size


class ResampleConfig:

    def __init__(
        self,
        original_sr: int = SampleRate.EXPECTED.value,
        target_sr: int = SampleRate.DEFAULT.value,
        res_type: ResampleType = ResampleType("polyphase"),
    ):
        self.original_sr = original_sr
        self.target_sr = target_sr
        self.res_type = res_type

    def resample_audio(self, audio: np.ndarray[float]) -> np.ndarray[float]:
        if self.original_sr == self.target_sr:
            return audio
        else:
            resampled_audio = librosa.resample(
                audio,
                orig_sr=self.original_sr,
                target_sr=self.target_sr,
                res_type=self.res_type,
            )
            return resampled_audio


class VadConfig:
    def __init__(self):

        self.onset_confirm_chunks: int = (
            10  # ~320ms of speech (n chunks * 32ms) to confirm speech onset
        )

        self.speech_complete_timeout_ms: int = (
            1024  # Timeout for speech completion in milliseconds
        )

        self.preroll_ring_size: int = (
            10  # Number of chunks to keep in the pre-roll buffer (~1 second of audio at 32ms per chunk)
        )

        self.max_utterance_seconds: int = (
            30  # Max Utterance Seconds for whisper is 30 seconds.
        )

        self.thread_timeout_seconds: int = (
            1  # Timeout for thread operations in seconds: `retry-loop poll interval`
        )

        try:
            assert (
                self.onset_confirm_chunks == self.preroll_ring_size
            ), "onset_confirm_chunks and preroll_ring_size must be equal for proper pre-roll functionality."

        except AssertionError as e:
            logger.warning(
                "AssertionError: {error}. Adjusting preroll_ring_size to match onset_confirm_chunks.",
                error=e,
            )
            self.preroll_ring_size = self.onset_confirm_chunks
