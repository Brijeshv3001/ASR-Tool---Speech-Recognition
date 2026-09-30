"""Evaluation module for computing Word Error Rate (WER) and Character Error Rate (CER)."""

import argparse
import logging
from typing import Dict, Union

logger = logging.getLogger("asr_tool.evaluate")


def evaluate_transcription(hypothesis: str, reference: str) -> Dict[str, float]:
    """Computes Word Error Rate (WER) and Character Error Rate (CER) between hypothesis and reference transcripts.

    Args:
        hypothesis: Model predicted transcript.
        reference: Ground truth reference transcript.

    Returns:
        Dict containing "wer" and "cer" float values (0.0 to 1.0+).
    """
    hyp_clean = hypothesis.strip()
    ref_clean = reference.strip()

    if not ref_clean:
        logger.warning("Reference text is empty.")
        return {"wer": 0.0 if not hyp_clean else 1.0, "cer": 0.0 if not hyp_clean else 1.0}

    try:
        import jiwer
        wer_score = float(jiwer.wer(ref_clean, hyp_clean))
        cer_score = float(jiwer.cer(ref_clean, hyp_clean))
        return {"wer": wer_score, "cer": cer_score}
    except Exception as e:
        logger.error("Error calculating WER/CER with jiwer: %s", e)
        # Basic fallback word-based calculation if jiwer fails
        ref_words = ref_clean.split()
        hyp_words = hyp_clean.split()
        if not ref_words:
            return {"wer": 0.0, "cer": 0.0}
        # Naive distance check fallback
        errs = abs(len(ref_words) - len(hyp_words))
        wer_fallback = float(errs) / float(len(ref_words))
        return {"wer": wer_fallback, "cer": wer_fallback}


def main():
    """CLI runner for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate ASR transcription against a reference text.")
    parser.add_argument("--hypothesis", required=True, help="Predicted transcript text or path to file")
    parser.add_argument("--reference", required=True, help="Reference transcript text or path to file")
    args = parser.parse_args()

    # Read from file if file paths provided
    hyp = args.hypothesis
    ref = args.reference

    try:
        with open(hyp, "r", encoding="utf-8") as f:
            hyp = f.read()
    except Exception:
        pass

    try:
        with open(ref, "r", encoding="utf-8") as f:
            ref = f.read()
    except Exception:
        pass

    metrics = evaluate_transcription(hyp, ref)
    print(f"Evaluation Results:")
    print(f"  Word Error Rate (WER)     : {metrics['wer'] * 100:.2f}%")
    print(f"  Character Error Rate (CER): {metrics['cer'] * 100:.2f}%")


if __name__ == "__main__":
    main()
