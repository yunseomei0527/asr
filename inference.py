#------------------------------
# ASR 추론(inference) 모듈
#------------------------------

import time
import librosa

from asr.audio_preprocess import preprocess_audio
from asr.model import WhisperASR
from asr.postprocess import normalize_text
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
