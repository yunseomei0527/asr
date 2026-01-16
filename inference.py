import time
import librosa

from asr.audio_preprocess import preprocess_audio
from asr.model import WhisperASR
from asr.postprocess import normalize_text

def run_asr(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(audio) / sr

    asr = WhisperASR()

    start = time.time()
    chunks = preprocess_audio(audio_path)
    raw_text = asr.transcribe(chunks)
    elapsed = time.time() - start

    final_text = normalize_text(raw_text)

    rtf = elapsed / audio_duration

    return final_text, rtf
