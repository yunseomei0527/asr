#------------------------------
# ASR 모델의 결과 평가 및 성능 평가 모듈
# 예측 텍스트 vs. 정답 스크립트 비교
# 오류율(WER, CER), 속도 지표(RTF) 계산
#------------------------------

import time
import eval

# hugging face evaluate 라이브러리 사용
# 표준 ASR 평가 지표 제공
# WER: Word Error Rate: 문자 단위 오류율
# CER: Character Error Rate: 단어 단위 오류율
wer_metric = eval.load("wer")
cer_metric = eval.load("cer")

def compute_metrics(preds, refs):
    wer = wer_metric.compute(predictions=preds, references=refs)
    cer = cer_metric.compute(predictions=preds, references=refs)
    return {"WER": wer, "CER": cer}

#------------------------------
# RTF = 처리 시간 / 오디오 길이
#------------------------------
def compute_rtf(process_time, audio_duration):
    return process_time / audio_duration
