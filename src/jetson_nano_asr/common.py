import librosa
import numpy as np
import sounddevice as sd

from enum import IntEnum
from pathlib import Path
from typing import Literal, NewType

from loggy import logger

BASE_URL = "http://localhost:8080/v1"

MAX_BUFFER_SIZE = 10  # Seconds?

DEFAULT_DEVICE_NAME = "webcam"  # Substring match against sounddevice's device names

WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large-v1, large-v2


def find_input_device(name_substring: str = DEFAULT_DEVICE_NAME) -> int:
    """
    Find the index of an input device whose name contains `name_substring`
    (case-insensitive).
    Parameters
    ----------
    name_substring : str
        Substring to search for in the device names. Default is "webcam".
    Returns
    -------
    int
        Index of the first matching input device. If no matching device is found, returns the index of the default input device. If multiple devices match, returns the first one that matches and is on the default host API.

    """
    devices = sd.query_devices()
    default_hostapi = sd.default.hostapi

    matches = [
        idx
        for idx, dev in enumerate(devices)
        if dev["max_input_channels"] > 0
        and name_substring.lower() in dev["name"].lower()
    ]

    for idx in matches:
        if devices[idx]["hostapi"] == default_hostapi:
            logger.info(
                "Selected input device {idx}: {name}",
                idx=idx,
                name=devices[idx]["name"],
            )
            return idx

    if matches:
        idx = matches[0]
        logger.info(
            "Selected input device {idx}: {name}", idx=idx, name=devices[idx]["name"]
        )
        return idx

    default_idx = sd.default.device[0]
    logger.warning(
        "No input device matching '{needle}' found. Falling back to system default input device {idx}: {name}",
        needle=name_substring,
        idx=default_idx,
        name=devices[default_idx]["name"],
    )
    return default_idx


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
        audio_device: int | None = None,
        mic: bool = True,
        file_path: Path | None = None,
        sample_rate: SampleRate = SampleRate.DEFAULT,
        channels: Channels = Channels.MONO,
        max_queue_size: int = 1024,
        utterance_max_queue_size: int = 10,
        speech_max_queue_size: int = 10,
    ):

        self.mic: bool = mic
        self.device: int | None = (
            audio_device
            if audio_device is not None
            else (find_input_device() if mic else None)
        )
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
            2048  # Timeout for speech completion in milliseconds
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
