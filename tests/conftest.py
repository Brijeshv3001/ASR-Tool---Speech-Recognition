"""Pytest conftest configuration and test fixtures for ASR-Tool."""

import os
import wave
import math
import struct
import tempfile
import pytest


def generate_sine_wav(filepath: str, duration_sec: float = 2.0, sample_rate: int = 16000) -> str:
    """Generates a simple 16kHz mono WAV file containing a sine audio tone.

    Args:
        filepath: Target output file path.
        duration_sec: Duration in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the created WAV file.
    """
    num_samples = int(duration_sec * sample_rate)
    frequency = 440.0  # Concert A (440 Hz)

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        for i in range(num_samples):
            # Generate sine wave sample (-32768 to 32767)
            t = float(i) / float(sample_rate)
            sample_val = int(16000.0 * math.sin(2.0 * math.pi * frequency * t))
            wf.writeframesraw(struct.pack('<h', sample_val))

    return filepath


@pytest.fixture(scope="session")
def sample_wav_path(tmp_path_factory) -> str:
    """Fixture providing path to a temporary valid 16kHz mono WAV file."""
    temp_dir = tmp_path_factory.mktemp("audio")
    wav_file = str(temp_dir / "test_sample.wav")
    return generate_sine_wav(wav_file, duration_sec=2.5, sample_rate=16000)


@pytest.fixture(scope="session", autouse=True)
def setup_samples_directory():
    """Autouse fixture ensuring samples/sample.wav exists for integration tests."""
    sample_wav = os.path.join("samples", "sample.wav")
    if not os.path.exists(sample_wav):
        generate_sine_wav(sample_wav, duration_sec=2.0, sample_rate=16000)
