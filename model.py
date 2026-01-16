#------------------------------
# audio_process.py에서 만든 chunck를 텍스트로 변환하는 ASR 모듈
#------------------------------

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class WhisperASR:
    #------------------------------
    # 모델 로딩 담당
    #------------------------------
    def __init__(self, model_name="openai/whisper-medium"):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(DEVICE)
        
    #------------------------------
    # 핵심 추론 함수
    #------------------------------
    def transcribe(self, audio_chunks, language="en"):
        texts = []

        for chunk in audio_chunks:
            # chunk 단위 처리 루프(whisper의 30초 입력 제한)
            inputs = self.processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt"
            )

            input_features = inputs.input_features.to(DEVICE)

            predicted_ids = self.model.generate(
                input_features,
                language=language,
                task="transcribe"
            )

            text = self.processor.batch_decode(
                # 토큰 ID -> 텍스트
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            #chunk 결과 병합
            texts.append(text)

        return " ".join(texts)
