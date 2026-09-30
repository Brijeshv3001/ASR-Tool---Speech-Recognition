"""Exporters for ASR-Tool results.

Supports exporting transcription results to Plain Text (.txt),
SubRip Subtitles (.srt), and Structured JSON (.json).
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger("asr_tool.exporters")


def format_timestamp(seconds: float) -> str:
    """Formats float seconds into SRT timestamp string format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like "00:01:23,456".
    """
    if seconds < 0:
        seconds = 0.0

    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3600000
    total_milliseconds %= 3600000
    minutes = total_milliseconds // 60000
    total_milliseconds %= 60000
    secs = total_milliseconds // 1000
    millis = total_milliseconds % 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def export_txt(result: Any, output_path: str) -> None:
    """Exports transcription result text to a plain text file.

    Args:
        result: TranscriptionResult object with .text attribute.
        output_path: Target output file path.
    """
    logger.info("Exporting TXT to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text.strip() + "\n")


def export_srt(result: Any, output_path: str) -> None:
    """Exports segment timestamps and text into SubRip Subtitles format (.srt).

    Args:
        result: TranscriptionResult object with .segments attribute.
        output_path: Target output file path.
    """
    logger.info("Exporting SRT to %s", output_path)
    lines = []
    segments = getattr(result, "segments", [])

    if not segments and hasattr(result, "text") and result.text:
        # Fallback if no segments provided
        start_str = format_timestamp(0.0)
        end_str = format_timestamp(getattr(result, "duration", 0.0))
        lines.append(f"1\n{start_str} --> {end_str}\n{result.text.strip()}\n")
    else:
        for idx, seg in enumerate(segments, start=1):
            start_str = format_timestamp(seg.start)
            end_str = format_timestamp(seg.end)
            lines.append(f"{idx}\n{start_str} --> {end_str}\n{seg.text.strip()}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def export_json(result: Any, output_path: str) -> None:
    """Exports complete transcription result metadata and segments to JSON format.

    Args:
        result: TranscriptionResult object.
        output_path: Target output file path.
    """
    logger.info("Exporting JSON to %s", output_path)
    segments_data = []
    for seg in getattr(result, "segments", []):
        segments_data.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip()
        })

    data: Dict[str, Any] = {
        "text": result.text.strip(),
        "language": getattr(result, "language", "auto"),
        "language_probability": round(getattr(result, "language_probability", 1.0), 4),
        "duration": round(getattr(result, "duration", 0.0), 3),
        "processing_time": round(getattr(result, "processing_time", 0.0), 3),
        "segments": segments_data
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
