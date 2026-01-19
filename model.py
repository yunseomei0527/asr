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
def transcribe(self, audio_chunks, language="ko"):
    texts = []

    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
        language=language,
        task="transcribe"
    )

    for chunk in audio_chunks:
        inputs = self.processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(DEVICE)

        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        texts.append(text)

    return " ".join(texts)

