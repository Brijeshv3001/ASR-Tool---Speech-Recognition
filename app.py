"""Streamlit Web Application for ASR-Tool.

Provides an intuitive, high-performance web interface for speech-to-text transcription,
microphone recording, timestamp inspection, and format exporting.
"""

import os
import tempfile
import time
import json
import streamlit as st

from asr_tool.transcriber import WhisperTranscriber, TranscriptionResult
from asr_tool.exporters import format_timestamp, export_txt, export_srt, export_json
from asr_tool.audio_utils import AudioProcessingError

# Page Configuration
st.set_page_config(
    page_title="ASR-Tool | Speech to Text",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Visual Aesthetics
st.markdown("""
<style>
    /* Global Styles & Dark Aesthetics */
    .main-header {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-lang {
        background-color: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .transcript-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #f8fafc;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_transcriber_model(model_size: str) -> WhisperTranscriber:
    """Loads and caches the WhisperTranscriber instance per model size.

    Args:
        model_size: Name of model size ("tiny", "base", "small", "medium").

    Returns:
        Loaded WhisperTranscriber instance.
    """
    return WhisperTranscriber(model_size=model_size)


def main():
    """Main Streamlit Application Layout."""

    # Sidebar Options
    with st.sidebar:
        st.image("https://img.icons8.com/isometric-folders/100/microphone.png", width=64)
        st.title("⚙️ Configurations")
        st.markdown("---")

        model_size = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium"],
            index=1,
            help="Base is fast and balanced. Medium provides maximum accuracy."
        )

        language_options = {
            "Auto-Detect": "auto",
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Chinese": "zh",
            "Japanese": "ja",
            "Hindi": "hi",
            "Tamil": "ta"
        }

        selected_lang_label = st.selectbox(
            "Target Language",
            options=list(language_options.keys()),
            index=0,
            help="Choose 'Auto-Detect' or lock to a specific spoken language."
        )
        selected_lang_code = language_options[selected_lang_label]

        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.info(
            "• **Tiny/Base**: Recommended for fast offline testing.\n"
            "• **Small/Medium**: Higher accuracy for complex audio.\n"
            "• Microphone input supported directly in the browser."
        )
        st.caption("ASR-Tool v1.0 • Offline University Project")

    # Header Section
    st.markdown('<div class="main-header">🎙️ ASR-Tool: Automatic Speech Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Offline audio-to-text transcription powered by OpenAI Whisper & faster-whisper.</div>',
        unsafe_allow_html=True
    )

    # Input Method Selection (Tabs)
    tab_upload, tab_mic = st.tabs(["📁 Upload Audio File", "🎤 Record Microphone"])

    audio_source_path = None
    input_file_name = "audio"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload an audio clip",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            help="Supported formats: WAV, MP3, M4A, FLAC, OGG"
        )
        if uploaded_file is not None:
            input_file_name = os.path.splitext(uploaded_file.name)[0]
            # Save uploaded file to temp path
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                audio_source_path = tmp.name

            st.audio(uploaded_file, format=uploaded_file.type)

    with tab_mic:
        st.write("Record speech directly from your microphone:")
        mic_audio = None
        if hasattr(st, "audio_input"):
            mic_audio = st.audio_input("Click to start recording")
        else:
            mic_audio = st.file_uploader("Record or select mic audio", type=["wav", "mp3", "ogg"], key="mic_fallback")

        if mic_audio is not None:
            input_file_name = "mic_recording"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(mic_audio.getvalue())
                audio_source_path = tmp.name

            st.audio(mic_audio)

    st.markdown("---")

    # Transcribe Button
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        transcribe_clicked = st.button("🚀 Transcribe Audio", type="primary", use_container_width=True)

    if transcribe_clicked:
        if not audio_source_path:
            st.warning("⚠️ Please upload an audio file or record from microphone first.")
            return

        try:
            with st.spinner(f"Loading '{model_size}' model & transcribing audio..."):
                transcriber = load_transcriber_model(model_size)
                result: TranscriptionResult = transcriber.transcribe(
                    audio_source_path,
                    language=selected_lang_code
                )
                st.session_state["last_result"] = result
                st.session_state["input_file_name"] = input_file_name

        except AudioProcessingError as ape:
            st.error(f"❌ Audio Processing Error: {ape}")
            return
        except Exception as ex:
            st.error(f"❌ Transcription Failure: {ex}")
            return

    # Render Results if Available
    if "last_result" in st.session_state:
        result: TranscriptionResult = st.session_state["last_result"]
        base_name = st.session_state.get("input_file_name", "transcription")

        st.subheader("📊 Transcription Metrics")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.duration:.2f}s</div>
                <div class="metric-label">Audio Duration</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.processing_time:.2f}s</div>
                <div class="metric-label">Processing Time</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.language.upper()}</div>
                <div class="metric-label">Detected Language</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.language_probability * 100:.1f}%</div>
                <div class="metric-label">Lang Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📝 Full Transcript")
        st.markdown(f'<div class="transcript-box">{result.text or "*(No speech detected)*"}</div>', unsafe_allow_html=True)

        # Tabbed View for Segment Timestamps and Exports
        tab_segments, tab_exports = st.tabs(["⏱️ Word / Segment Timestamps", "💾 Export Formats"])

        with tab_segments:
            if result.segments:
                table_data = [
                    {
                        "Segment #": i + 1,
                        "Start Time": format_timestamp(s.start),
                        "End Time": format_timestamp(s.end),
                        "Duration (s)": round(s.end - s.start, 2),
                        "Text": s.text
                    }
                    for i, s in enumerate(result.segments)
                ]
                st.dataframe(table_data, use_container_width=True)
            else:
                st.info("No timed segments recorded.")

        with tab_exports:
            st.write("Download transcription in your preferred format:")

            # Prepare export contents
            txt_content = result.text.strip() + "\n"

            # Prepare SRT content
            srt_lines = []
            for idx, seg in enumerate(result.segments, start=1):
                s_str = format_timestamp(seg.start)
                e_str = format_timestamp(seg.end)
                srt_lines.append(f"{idx}\n{s_str} --> {e_str}\n{seg.text}\n")
            srt_content = "\n".join(srt_lines).strip() + "\n"

            # Prepare JSON content
            json_data = {
                "text": result.text.strip(),
                "language": result.language,
                "language_probability": round(result.language_probability, 4),
                "duration": round(result.duration, 3),
                "processing_time": round(result.processing_time, 3),
                "segments": [
                    {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                    for s in result.segments
                ]
            }
            json_content = json.dumps(json_data, indent=2, ensure_ascii=False)

            e1, e2, e3 = st.columns(3)
            with e1:
                st.download_button(
                    "📄 Download .TXT",
                    data=txt_content,
                    file_name=f"{base_name}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with e2:
                st.download_button(
                    "🎬 Download .SRT",
                    data=srt_content,
                    file_name=f"{base_name}.srt",
                    mime="application/x-subrip",
                    use_container_width=True
                )
            with e3:
                st.download_button(
                    "📊 Download .JSON",
                    data=json_content,
                    file_name=f"{base_name}.json",
                    mime="application/json",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
