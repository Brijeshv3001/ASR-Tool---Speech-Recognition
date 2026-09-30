"""Integration test for WhisperTranscriber."""

import pytest
from asr_tool.transcriber import WhisperTranscriber, TranscriptionResult, Segment


def test_transcriber_invalid_model():
    """Test initialization failure with invalid model size."""
    with pytest.raises(ValueError):
        WhisperTranscriber(model_size="invalid_size_xyz")


@pytest.mark.integration
def test_transcriber_integration_tiny(sample_wav_path):
    """Integration test: transcribes a short WAV sample clip with the 'tiny' Whisper model."""
    transcriber = WhisperTranscriber(model_size="tiny")
    result = transcriber.transcribe(sample_wav_path)

    assert isinstance(result, TranscriptionResult)
    assert result.duration > 0
    assert result.processing_time > 0
    assert isinstance(result.segments, list)
    assert isinstance(result.language, str)
