import numpy as np
from enum import IntEnum
from pathlib import Path
from typing import Literal, NewType

import librosa

MAX_BUFFER_SIZE = 10  # Seconds?

DEVICE = 14

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
        device: int = DEVICE,
        file_path: Path | None = None,  # Path(__file__).parent / "test.wav",
        device_name: str = "Full HD webcam",
        sample_rate: SampleRate = SampleRate.DEFAULT,
        channels: Channels = Channels.MONO,
        chunk_size: int = 1536,  # 512 * 3 (downsample from 48kHz to 16kHz)
        max_queue_size: int = 1024,
    ):
        self.device: int = device
        self.file_path: Path | None = file_path
        self.device_name: str = device_name
        self.sample_rate: SampleRate = sample_rate
        self.channels: Channels = channels
        self.chunk_size: int = chunk_size
        self.max_queue_size: int = max_queue_size


class ResampleConfig:

    def __init__(
        self,
        original_sr: int = SampleRate.EXPECTED.value,
        target_sr: int = SampleRate.DEFAULT.value,
        res_type: ResampleType = "polyphase",
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
