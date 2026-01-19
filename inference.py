#------------------------------
# ASR 추론(inference) 모듈
#------------------------------

import time
import librosa

from asr.audio_preprocess import preprocess_audio, load_reference_text
from asr.model import WhisperASR
from asr.postprocess import normalize_text
from asr.metrics import compute_metrics

'''
def run_asr(audio_path):
    # 전처리 전 원본 오디오 길이 계산(RTF 계산용)
    audio, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(audio) / sr
    
    # ASR 모델 로드
    asr = WhisperASR()
    
    # 추론 시간 측정
    start = time.time()
    # 오디오 전처리
    chunks = preprocess_audio(audio_path)
    # ASR 추론
    raw_text = asr.transcribe(chunks)
    elapsed = time.time() - start

    # 텍스트 후처리
    final_text = normalize_text(raw_text)
    # RTF 계산
    rtf = elapsed / audio_duration

    return final_text, rtf
'''
'''
def run_asr(audio_path):
    # 전처리 전 원본 오디오 길이 계산(RTF 계산용)
    audio, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(audio) / sr
    
    # ASR 모델 로드
    asr = WhisperASR()
    # 추론 시간 측정
    start = time.time()
    # 오디오 전처리
    chunks = preprocess_audio(audio_path)

    chunk_texts = []
    for i, chunk in enumerate(chunks):
        text = asr.transcribe([chunk])
        chunk_texts.append(text)

    elapsed = time.time() - start
    rtf = elapsed / audio_duration

    return chunk_texts, rtf
'''



def run_asr_with_eval(audio_path, transcript_json_path):
    # ----------------------
    # 오디오 길이 (RTF)
    # ----------------------
    audio, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(audio) / sr

    # ----------------------
    # ASR 모델
    # ----------------------
    asr = WhisperASR()

    # ----------------------
    # ASR 추론
    # ----------------------
    start = time.time()
    chunks = preprocess_audio(audio_path)

    pred_chunks = []
    for chunk in chunks:
        text = asr.transcribe([chunk])
        pred_chunks.append(text)

    elapsed = time.time() - start
    rtf = elapsed / audio_duration

    pred_text = normalize_text(" ".join(pred_chunks))

    # ----------------------
    # 정답 자막 로드
    # ----------------------
    ref_text = load_reference_text(transcript_json_path)

    # ----------------------
    # 평가 지표
    # ----------------------
    metrics = compute_metrics(
        preds=[pred_text],
        refs=[ref_text]
    )
    metrics["RTF"] = rtf

    return pred_text, ref_text, metrics
