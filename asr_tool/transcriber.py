"""Core Whisper Transcriber module for ASR-Tool using faster-whisper."""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from asr_tool.audio_utils import (
    convert_to_wav_16k_mono,
    load_audio_as_numpy,
    get_audio_duration,
    AudioProcessingError
)

logger = logging.getLogger("asr_tool.transcriber")


@dataclass
class Segment:
    """Dataclass representing a timestamped segment of transcription."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Dataclass storing full transcription output and metadata."""
    text: str
    segments: List[Segment] = field(default_factory=list)
    language: str = "en"
    language_probability: float = 1.0
    duration: float = 0.0
    processing_time: float = 0.0


class ModelDownloadError(Exception):
    """Raised when faster-whisper fails to download or initialize the model."""
    pass


class WhisperTranscriber:
    """Class wrapper for faster-whisper model loading and speech transcription."""

    VALID_MODELS = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default"
    ) -> None:
        """Initializes and loads the faster-whisper model once.

        Args:
            model_size: Size of Whisper model (tiny, base, small, medium).
            device: Device type ("auto", "cpu", "cuda").
            compute_type: Quantization compute type ("default", "int8", "float16").

        Raises:
            ValueError: If model size is invalid.
            ModelDownloadError: If model fails to load.
        """
        if model_size.lower() not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid model size '{model_size}'. Valid sizes: {', '.join(sorted(self.VALID_MODELS))}"
            )

        self.model_size = model_size.lower()
        self.device = device
        self.compute_type = compute_type
        self.model = None

        logger.info(
            "Loading faster-whisper model '%s' (device: %s, compute_type: %s)...",
            self.model_size,
            self.device,
            self.compute_type
        )
        self._load_model()

    def _load_model(self) -> None:
        """Loads the faster-whisper model instance."""
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Model '%s' successfully loaded.", self.model_size)
        except Exception as e:
            logger.error("Failed to load faster-whisper model '%s': %s", self.model_size, e)
            raise ModelDownloadError(
                f"Could not load faster-whisper model '{self.model_size}'. Check network connection or model name."
            ) from e

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribes an audio file into text with timestamps.

        Args:
            audio_path: Path to input audio file.
            language: Language code (e.g., "en", "es", "fr") or None / "auto" for auto-detection.

        Returns:
            TranscriptionResult containing transcript text, segments, and performance metadata.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Preprocess audio to 16kHz mono WAV
        wav_path = convert_to_wav_16k_mono(audio_path)
        audio_duration = get_audio_duration(wav_path)
        audio_array = load_audio_as_numpy(wav_path)

        lang_param = None if (not language or language.lower() in ("auto", "auto-detect")) else language.lower()

        logger.info(
            "Starting transcription for %s (duration: %.2fs, language: %s)...",
            audio_path,
            audio_duration,
            lang_param or "auto-detect"
        )

        start_time = time.perf_counter()

        try:
            # Pass numpy float32 array directly to model.transcribe to avoid PyAV open() version incompatibilities
            segments_gen, info = self.model.transcribe(
                audio_array,
                language=lang_param,
                beam_size=5,
                word_timestamps=True
            )

            segment_list: List[Segment] = []
            text_parts: List[str] = []

            for seg in segments_gen:
                clean_text = seg.text.strip()
                if clean_text:
                    segment_list.append(Segment(
                        start=float(seg.start),
                        end=float(seg.end),
                        text=clean_text
                    ))
                    text_parts.append(clean_text)

            processing_time = time.perf_counter() - start_time
            full_text = " ".join(text_parts)

            detected_lang = getattr(info, "language", lang_param or "en")
            lang_prob = float(getattr(info, "language_probability", 1.0))

            logger.info(
                "Transcription finished in %.2fs. Detected language: %s (prob: %.2f)",
                processing_time,
                detected_lang,
                lang_prob
            )

            return TranscriptionResult(
                text=full_text,
                segments=segment_list,
                language=detected_lang,
                language_probability=lang_prob,
                duration=audio_duration,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            raise AudioProcessingError(f"Error transcribing audio file '{audio_path}': {e}") from e
