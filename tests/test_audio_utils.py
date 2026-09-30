"""Unit tests for audio_utils module."""

import os
import pytest
from asr_tool.audio_utils import (
    validate_audio_file,
    convert_to_wav_16k_mono,
    get_audio_duration,
    UnsupportedFormatError,
    CorruptAudioError,
    find_ffmpeg_executable
)


def test_find_ffmpeg_executable():
    """Verify ffmpeg executable detection."""
    exe = find_ffmpeg_executable()
    assert exe is not None, "FFmpeg executable (or imageio-ffmpeg) should be found."
    assert os.path.exists(exe)


def test_validate_audio_file_nonexistent():
    """Test validation failure on nonexistent file."""
    with pytest.raises(FileNotFoundError):
        validate_audio_file("non_existent_file.wav")


def test_validate_audio_file_empty(tmp_path):
    """Test validation failure on 0-byte file."""
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(CorruptAudioError):
        validate_audio_file(str(empty_file))


def test_validate_audio_file_unsupported(tmp_path):
    """Test validation failure on unsupported extension."""
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("hello world")
    with pytest.raises(UnsupportedFormatError):
        validate_audio_file(str(invalid_file))


def test_validate_audio_file_success(sample_wav_path):
    """Test validation success on valid WAV file."""
    validate_audio_file(sample_wav_path)


def test_get_audio_duration(sample_wav_path):
    """Test accurate audio duration calculation."""
    duration = get_audio_duration(sample_wav_path)
    assert isinstance(duration, float)
    assert 2.0 <= duration <= 3.0


def test_convert_to_wav_16k_mono(sample_wav_path, tmp_path):
    """Test audio conversion to 16kHz mono WAV."""
    out_wav = str(tmp_path / "converted.wav")
    result_path = convert_to_wav_16k_mono(sample_wav_path, out_wav)

    assert os.path.exists(result_path)
    assert get_audio_duration(result_path) > 0
