import modal
import os
import subprocess
import time
from pathlib import Path
from modal import enter, method

# Modal Volume 생성 및 마운트 경로 지정
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")
MODEL_ID = "gemma:7b"

# Ollama 서비스를 시작하고 gemma:7b와 nomic-embed-text 모델 가중치를 Modal Volume에 저장하는 함수
def pull_model_weights():
    # 시스템 데몬 재로드 및 ollama 서비스 활성화
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    time.sleep(5)  # ollama 서비스 기동 대기

    # gemma:7b 모델 가중치 다운로드 (이미 다운로드되어 있지 않다면)
    subprocess.run(["ollama", "pull", MODEL_ID], stdout=subprocess.PIPE)
    # nomic-embed-text 모델 가중치 다운로드
    subprocess.run(["ollama", "pull", "nomic-embed-text"], stdout=subprocess.PIPE)



# 이미지 빌드: 필요한 의존성 설치, ollama 설치 및 pull_model_weights 실행
image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .add_local_file("ollama_modal/requirements.txt", "/etc/systemd/requirements.txt", copy=True)
    .add_local_file("ollama_modal/ollama.service", "/etc/systemd/system/ollama.service", copy=True)
    .run_commands(
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
        "pip install -r /etc/systemd/requirements.txt",
    )
    .pip_install("ollama")
    .run_function(pull_model_weights)  # 빌드 시 한번 실행하여 모델 가중치를 다운로드
)



app = modal.App(name="ollama", image=image)

with image.imports():
    import ollama
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import ChatPromptTemplate

# Gemma:7b 모델 가중치를 Modal Volume에서 로드하고, 추론 시 재사용하는 클래스
@app.cls(gpu="a10g", volumes={MODEL_DIR: volume}, scaledown_window=300)
class GemmaModel:
    @enter()
    def setup(self):
        """
        컨테이너가 시작될 때 한 번 실행되어,
        - ollama 서비스를 시작하고,
        - MODEL_DIR 내에 gemma:7b 가중치가 없는 경우 다운로드하며,
        - ChatOllama를 통해 모델을 메모리에 로드합니다.
        """
        # ollama 서비스 기동
        subprocess.run(["systemctl", "start", "ollama"])
        model_path = MODEL_DIR / MODEL_ID
        if not model_path.exists():
            subprocess.run(["ollama", "pull", MODEL_ID], stdout=subprocess.PIPE)
        # 모델 가중치는 볼륨에 저장되어 있으므로,
        # 이후 추론 시에는 캐시된 모델을 사용하도록 합니다.
        self.llm = ChatOllama(model=MODEL_ID)

    @method(is_generator=True)
    def chain_ollama(self, topic: str):
        """
        주어진 주제에 대해 간략한 설명을 생성합니다.
        컨텍스트 매니저를 통해 한 번만 모델 가중치를 로드한 후,
        추론 요청 시 저장된 모델 인스턴스를 사용합니다.
        """
        prompt = ChatPromptTemplate.from_template("{topic}에 대하여 간략히 설명해 줘.")
        chain = prompt | self.llm | StrOutputParser()
        yield from chain.stream({"topic": topic})

# 로컬 엔트리포인트: GemmaModel 인스턴스를 생성하고 추론을 실행합니다.
@app.local_entrypoint()
def main(topic: str = "서울", model: str = "gemma:7b", lookup: bool = False):
    """
    GemmaModel을 사용하여 Modal Volume에 저장된 gemma:7b 모델 가중치로 추론을 실행합니다.
    """
    if lookup:
        gemma = modal.Cls.lookup("ollama", "GemmaModel")
    else:
        gemma = GemmaModel()

    if topic:
        for chunk in gemma.chain_ollama.remote_gen(topic):
            print(chunk, end="", flush=True)
