"""Unit tests for evaluate module."""

from asr_tool.evaluate import evaluate_transcription


def test_evaluate_perfect_match():
    """Test WER and CER evaluation when hypothesis matches reference exactly."""
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumps over the lazy dog"

    metrics = evaluate_transcription(hyp, ref)
    assert metrics["wer"] == 0.0
    assert metrics["cer"] == 0.0


def test_evaluate_partial_match():
    """Test WER calculation with substitutions/deletions."""
    ref = "the quick brown fox"
    hyp = "the fast brown fox"

    metrics = evaluate_transcription(hyp, ref)
    assert metrics["wer"] > 0.0
    assert metrics["wer"] <= 0.5
    assert metrics["cer"] > 0.0


def test_evaluate_empty_reference():
    """Test handle empty reference text."""
    metrics = evaluate_transcription("test transcript", "")
    assert metrics["wer"] == 1.0
    assert metrics["cer"] == 1.0


def test_evaluate_empty_hypothesis():
    """Test handle empty hypothesis text."""
    metrics = evaluate_transcription("", "reference transcript")
    assert metrics["wer"] == 1.0
    assert metrics["cer"] == 1.0
