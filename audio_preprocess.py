#------------------------------
# 원본 오디오 파일을 ASR(음성 인식) 모델에 적합한 형식으로 전처리하는 모듈
#------------------------------

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000
MAX_CHUNK_SEC = 30
OVERLAP_SEC = 3

#------------------------------
# 오디오 파일 로드 함수
#------------------------------
def load_audio(path):
    audio, sr = librosa.load(path, sr=None, mono=False)

    # stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # resample
    # whisper 모델의 입력 조건에 맞추기 위함
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    return audio, TARGET_SR

#------------------------------
# 무음 구간 제거 함수
#------------------------------
def trim_silence(audio, sr):
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=20
        # 20db 이하 구간을 무음으로 간주
    )
    return trimmed

#------------------------------
# 오디오 볼륨 정규화 함수(RMS 기준)
#------------------------------
def normalize_volume(audio):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        audio = audio / rms
    return audio

#------------------------------
# 30초 단위로 분할, 3초 겹치기 함수
# wisper는 약 30초 입력을 기준으로 학습됨
# ex. [0-30초][27-57초][54-84초]
#------------------------------
def split_audio(audio, sr):
    chunk_len = MAX_CHUNK_SEC * sr
    overlap_len = OVERLAP_SEC * sr

    chunks = []
    start = 0

    while start < len(audio):
        end = start + chunk_len
        chunk = audio[start:end]
        chunks.append(chunk)
        start = end - overlap_len

    return chunks

#------------------------------
# 전처리 함수: 오디오 로드 -> 무음 제거 -> 볼륨 정규화 -> 30초 단위 분할
#------------------------------
def preprocess_audio(path):
    audio, sr = load_audio(path)
    audio = trim_silence(audio, sr)
    audio = normalize_volume(audio)
    chunks = split_audio(audio, sr)
    return chunks
