"""Audio utilities for ASR-Tool.

Handles audio format validation, conversion to 16kHz mono WAV, numpy array conversion, and duration calculation.
"""

import os
import shutil
import logging
import wave
import contextlib
from typing import Optional
import numpy as np

logger = logging.getLogger("asr_tool.audio_utils")

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

class AudioProcessingError(Exception):
    """Base exception for audio processing errors."""
    pass

class UnsupportedFormatError(AudioProcessingError):
    """Raised when the audio format is not supported."""
    pass

class MissingFFmpegError(AudioProcessingError):
    """Raised when FFmpeg executable is not found."""
    pass

class CorruptAudioError(AudioProcessingError):
    """Raised when audio file is corrupt or unreadable."""
    pass


def find_ffmpeg_executable() -> Optional[str]:
    """Find FFmpeg executable in PATH or via imageio_ffmpeg if installed.

    Returns:
        Path to ffmpeg executable or None if not found.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_path):
            return ffmpeg_path
    except ImportError:
        pass

    return None


def setup_pydub_ffmpeg() -> None:
    """Configures pydub AudioSegment to use imageio-ffmpeg if system ffmpeg is missing."""
    ffmpeg_exe = find_ffmpeg_executable()
    if ffmpeg_exe:
        try:
            from pydub import AudioSegment
            AudioSegment.converter = ffmpeg_exe
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            ffprobe_exe = shutil.which("ffprobe", path=ffmpeg_dir)
            if ffprobe_exe:
                AudioSegment.ffprobe = ffprobe_exe
        except ImportError:
            pass


def validate_audio_file(file_path: str) -> None:
    """Validates that audio file exists, is non-empty, and has supported extension.

    Args:
        file_path: Absolute or relative path to the audio file.

    Raises:
        FileNotFoundError: If the file does not exist.
        CorruptAudioError: If the file is empty (0 bytes).
        UnsupportedFormatError: If file extension is not supported.
    """
    if not os.path.exists(file_path):
        logger.error("Audio file does not exist: %s", file_path)
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if os.path.getsize(file_path) == 0:
        logger.error("Audio file is empty (0 bytes): %s", file_path)
        raise CorruptAudioError(f"Audio file is empty (0 bytes): {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported audio format: %s", ext)
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def convert_to_wav_16k_mono(input_path: str, output_path: Optional[str] = None) -> str:
    """Converts audio file to 16kHz mono WAV format suitable for Whisper inference.

    Args:
        input_path: Path to input audio file.
        output_path: Optional output path. If None, creates a file in temporary/same folder.

    Returns:
        Path to the converted 16kHz mono WAV file.

    Raises:
        AudioProcessingError: On conversion failure or missing FFmpeg.
    """
    validate_audio_file(input_path)

    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_16k.wav"

    # Fast path: check if file is already valid 16kHz mono WAV
    if input_path.lower().endswith(".wav"):
        try:
            with contextlib.closing(wave.open(input_path, 'r')) as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                    logger.info("Input file is already valid 16kHz mono WAV: %s", input_path)
                    return input_path
        except Exception:
            pass

    setup_pydub_ffmpeg()
    try:
        from pydub import AudioSegment
        logger.info("Converting audio with pydub: %s -> %s", input_path, output_path)
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        logger.info("Successfully converted audio to 16kHz mono WAV: %s", output_path)
        return output_path
    except Exception as e:
        ffmpeg_exe = find_ffmpeg_executable()
        if not ffmpeg_exe:
            raise MissingFFmpegError(
                "FFmpeg executable not found. Please install FFmpeg or imageio-ffmpeg package."
            ) from e
        raise AudioProcessingError(f"Failed to process audio file '{input_path}': {e}") from e


def load_audio_as_numpy(wav_path: str) -> np.ndarray:
    """Loads a 16kHz mono WAV audio file into a 1D float32 numpy array [-1.0, 1.0].

    Args:
        wav_path: Path to 16kHz mono WAV file.

    Returns:
        1D numpy array of float32 samples.
    """
    try:
        with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_bytes = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        return audio
    except Exception as e:
        setup_pydub_ffmpeg()
        from pydub import AudioSegment
        audio_seg = AudioSegment.from_file(wav_path).set_frame_rate(16000).set_channels(1)
        samples = audio_seg.get_array_of_samples()
        max_val = float(1 << (audio_seg.sample_width * 8 - 1))
        return np.array(samples, dtype=np.float32) / max_val


def get_audio_duration(file_path: str) -> float:
    """Calculates total audio duration in seconds.

    Args:
        file_path: Path to audio file.

    Returns:
        Audio duration in seconds.
    """
    validate_audio_file(file_path)

    if file_path.lower().endswith(".wav"):
        try:
            with contextlib.closing(wave.open(file_path, 'r')) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return float(frames) / float(rate)
        except Exception:
            pass

    setup_pydub_ffmpeg()
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        return float(len(audio)) / 1000.0
    except Exception as e:
        logger.error("Failed to determine audio duration for %s: %s", file_path, e)
        raise CorruptAudioError(f"Could not read audio duration for '{file_path}': {e}") from e
