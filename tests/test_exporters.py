"""Unit tests for exporters module."""

import os
import json
import pytest

from asr_tool.exporters import format_timestamp, export_txt, export_srt, export_json
from asr_tool.transcriber import Segment, TranscriptionResult


def test_format_timestamp():
    """Test SRT timestamp formatting (HH:MM:SS,mmm)."""
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(1.5) == "00:00:01,500"
    assert format_timestamp(65.432) == "00:01:05,432"
    assert format_timestamp(3661.123) == "01:01:01,123"


def test_export_txt(tmp_path):
    """Test export_txt writes transcript correctly."""
    result = TranscriptionResult(text="Hello world speech test.")
    out_file = str(tmp_path / "output.txt")

    export_txt(result, out_file)

    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    assert content == "Hello world speech test."


def test_export_srt(tmp_path):
    """Test export_srt writes valid SRT format with timestamps."""
    segments = [
        Segment(start=0.0, end=1.5, text="Hello world."),
        Segment(start=1.6, end=3.0, text="This is a test.")
    ]
    result = TranscriptionResult(text="Hello world. This is a test.", segments=segments)
    out_file = str(tmp_path / "output.srt")

    export_srt(result, out_file)

    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "1\n00:00:00,000 --> 00:00:01,500\nHello world." in content
    assert "2\n00:00:01,600 --> 00:00:03,000\nThis is a test." in content


def test_export_json(tmp_path):
    """Test export_json writes structured metadata and segments."""
    segments = [Segment(start=0.0, end=1.5, text="Hello world.")]
    result = TranscriptionResult(
        text="Hello world.",
        segments=segments,
        language="en",
        language_probability=0.98,
        duration=2.5,
        processing_time=0.45
    )
    out_file = str(tmp_path / "output.json")

    export_json(result, out_file)

    assert os.path.exists(out_file)
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["text"] == "Hello world."
    assert data["language"] == "en"
    assert data["language_probability"] == 0.98
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "Hello world."
