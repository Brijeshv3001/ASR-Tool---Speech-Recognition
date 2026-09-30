"""Command Line Interface (CLI) for ASR-Tool."""

import sys
import os
import argparse
import logging
from typing import Optional

from asr_tool.transcriber import WhisperTranscriber, AudioProcessingError, ModelDownloadError
from asr_tool.exporters import export_txt, export_srt, export_json
from asr_tool.evaluate import evaluate_transcription

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("asr_tool.cli")


def build_parser() -> argparse.ArgumentParser:
    """Builds argparse CLI command structure.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="asr_tool",
        description="ASR-Tool: Automatic Speech Recognition CLI (Offline Whisper Transcription & Evaluation)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Transcribe subcommand
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio file to text")
    transcribe_parser.add_argument("audio", help="Path to input audio file (.wav, .mp3, .m4a, .flac, .ogg)")
    transcribe_parser.add_argument(
        "--model", "-m",
        default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: base)"
    )
    transcribe_parser.add_argument(
        "--language", "-l",
        default="auto",
        help="Language code (e.g., en, es, fr) or 'auto' for auto-detection (default: auto)"
    )
    transcribe_parser.add_argument(
        "--format", "-f",
        default="txt",
        choices=["txt", "srt", "json", "all"],
        help="Output export format (default: txt)"
    )
    transcribe_parser.add_argument(
        "--output", "-o",
        help="Path for output file (or output directory if format is 'all')"
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Compute WER and CER between predicted transcript and reference")
    eval_parser.add_argument("--hypothesis", "-hyp", required=True, help="Predicted transcript text or path to file")
    eval_parser.add_argument("--reference", "-ref", required=True, help="Ground truth reference transcript text or file")
    eval_parser.add_argument("--output", "-o", help="Optional output JSON path for evaluation metrics")

    return parser


def handle_transcribe(args: argparse.Namespace) -> int:
    """Handles transcribe command execution.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    audio_path = args.audio
    if not os.path.exists(audio_path):
        logger.error("Input audio file not found: %s", audio_path)
        return 1

    try:
        transcriber = WhisperTranscriber(model_size=args.model)
        result = transcriber.transcribe(audio_path, language=args.language)

        print("\n" + "=" * 60)
        print(" TRANSCRIPTION COMPLETED SUCCESSFUL")
        print("=" * 60)
        print(f" Audio File       : {audio_path}")
        print(f" Audio Duration   : {result.duration:.2f} s")
        print(f" Processing Time  : {result.processing_time:.2f} s")
        print(f" Language Detected: {result.language} (Confidence: {result.language_probability * 100:.1f}%)")
        print("-" * 60)
        print(" TRANSCRIPT PREVIEW:")
        print(f" \"{result.text.strip()}\"")
        print("=" * 60 + "\n")

        fmt = args.format.lower()
        out_target = args.output

        if fmt == "all":
            base_path = out_target if out_target else os.path.splitext(audio_path)[0]
            if os.path.isdir(base_path):
                base = os.path.join(base_path, os.path.basename(os.path.splitext(audio_path)[0]))
            else:
                base = os.path.splitext(base_path)[0]

            export_txt(result, f"{base}.txt")
            export_srt(result, f"{base}.srt")
            export_json(result, f"{base}.json")
            print(f"Exported all formats to: {base}.txt, {base}.srt, {base}.json")

        elif fmt == "srt":
            out_file = out_target or f"{os.path.splitext(audio_path)[0]}.srt"
            export_srt(result, out_file)
            print(f"Exported SRT to: {out_file}")

        elif fmt == "json":
            out_file = out_target or f"{os.path.splitext(audio_path)[0]}.json"
            export_json(result, out_file)
            print(f"Exported JSON to: {out_file}")

        else: # txt
            out_file = out_target or f"{os.path.splitext(audio_path)[0]}.txt"
            export_txt(result, out_file)
            print(f"Exported Text to: {out_file}")

        return 0

    except (AudioProcessingError, ModelDownloadError, ValueError) as e:
        logger.error("Transcription error: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during CLI transcription: %s", e)
        return 1


def handle_evaluate(args: argparse.Namespace) -> int:
    """Handles evaluate command execution.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    hyp = args.hypothesis
    ref = args.reference

    if os.path.exists(hyp):
        with open(hyp, "r", encoding="utf-8") as f:
            hyp = f.read()

    if os.path.exists(ref):
        with open(ref, "r", encoding="utf-8") as f:
            ref = f.read()

    metrics = evaluate_transcription(hyp, ref)
    print("\n" + "=" * 50)
    print(" EVALUATION RESULTS (jiwer)")
    print("=" * 50)
    print(f" Word Error Rate (WER)     : {metrics['wer'] * 100:.2f}% ({metrics['wer']:.4f})")
    print(f" Character Error Rate (CER): {metrics['cer'] * 100:.2f}% ({metrics['cer']:.4f})")
    print("=" * 50 + "\n")

    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to: {args.output}")

    return 0


def main():
    """Main CLI entrypoint."""
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "transcribe":
        sys.exit(handle_transcribe(args))
    elif args.command == "evaluate":
        sys.exit(handle_evaluate(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
