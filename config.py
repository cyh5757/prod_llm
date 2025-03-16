from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
    """
    Modal 실행을 위한 기본 설정 클래스
    """

    # Modal을 사용할지 여부 (True: 사용, False: 사용하지 않음)
    use_modal = True

    # 기본 GPU 타입 설정
    # A10G는 중간 성능의 GPU로 추론 및 학습에 적합
    gpu_type = "a10g"

    # 다중 화자 음성 분리(Speaker Diarization)용 GPU 설정
    # T4는 비용 효율적인 GPU로 적절한 성능 제공
    gpu_type_diarization = "t4"

    # 모델 캐시를 위한 볼륨 이름 설정 (중복 다운로드 방지)
    model_cache_volume = "model-cache"

    # 사용할 Ollama 모델 (기본값: "gemma:7b")
    model_name = "gemma:7b"

    # Modal 앱 이름
    app_name = "ollama"

