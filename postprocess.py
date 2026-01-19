#------------------------------
# whisper가 출력한 원시 텍스트 후처리 모듈
#------------------------------

import re

# 말버릇 목록, 의미 없는 발화 제거
FILLERS = []

def normalize_text(text):
    # 소문자 통일
    text = text.lower()

    # 숫자 표기 단순 정규화 (확장 가능)
    text = re.sub(r"\bzero\b", "0", text)

    # 특수기호 통일
    text = re.sub(r"[“”]", "\"", text)
    text = re.sub(r"[’]", "'", text)

    # filler 제거 (옵션)
    for f in FILLERS:
        text = re.sub(rf"\b{f}\b", "", text)

    # 중복 공백 제거
    text = re.sub(r"\s+", " ", text)

    return text.strip()
