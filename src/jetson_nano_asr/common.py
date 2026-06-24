from enum import IntEnum 
from pathlib import Path    
from typing import Literal

import librosa 

MAX_BUFFER_SIZE = 10 # Seconds?

DEVICES = (14, 17)

class SampleRate(IntEnum):
    EXPECTED: int = 48000 
    DEFAULT: int = 16_000
    HALF_SUPPORT: int = 8_000

class Channels(IntEnum):
    MONO: int = 1
    STEREO: int = 2
    SURROUND: int = 6

    
class RecordingConfig:
    device: tuple[int, int]= DEVICES
    file_path: Path | None = Path(__file__).parent / "test.wav"
    device_name: str = "Full HD webcam"
    sample_rate:SampleRate = SampleRate.EXPECTED
    channels:Channels = Channels.MONO
    chunk_size:int = 512 
    

class ResampleConfig:

    original_sr = SampleRate.EXPECTED
    target_sr = SampleRate.DEFAULT # Or can be set to SampleRate.HALF_SUPPORT for quicker processing
    res_type: Literal["kaiser_best", "kaiser_fast", "fft", "polyphase", "soxr_qq", "soxr_vhq", "soxr_hq", "soxr_mq", "soxr_lq"] = "kaiser_best"


    @classmethod
    def resample_audio(cls, audio: list[float]) -> list[float]:
        if cls.original_sr == cls.target_sr:
            return audio
        else:
            resampled_audio = librosa.resample(audio, orig_sr=cls.original_sr, target_sr=cls.target_sr, res_type=cls.res_type)
            return resampled_audio
    