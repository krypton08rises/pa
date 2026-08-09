from .asr import main as stream_captions
from pathlib import Path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio using Whisper model."
    )

    parser.add_argument(
        "--mic-mode",
        action="store_true",
        help="Use microphone input instead of a file.",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Path to the audio file to transcribe (used when --mic-mode is not set).",
    )

    args = parser.parse_args()

    if not args.mic_mode and not args.file_path:
        args.file_path = Path(__file__).parent / "data/test_audios/out.wav"

    stream_captions(
        mic_mode=args.mic_mode,
        file_path=Path(args.file_path) if args.file_path else None,
    )
