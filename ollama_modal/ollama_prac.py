import modal
import os
import subprocess
import time

from modal import enter, method

# 사용 가능한 모델 목록
DEFAULT_MODELS = ["gemma:7b"]
MODEL = os.environ.get("MODEL", "gemma:7b")  # 기본 모델을 gemma:7b로 설정


def pull():
    """Ollama 서비스 실행 및 모델 다운로드"""
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    time.sleep(5)  # 서버 실행 대기

    # 모든 모델 다운로드
    for model in DEFAULT_MODELS:
        subprocess.run(["ollama", "pull", model], stdout=subprocess.PIPE)


image = (
    modal.Image.debian_slim()
    .apt_install("curl", "systemctl")
    .add_local_file("requirements.txt", "/etc/systemd/requirements.txt", copy=True)
    .add_local_file("ollama.service", "/etc/systemd/system/ollama.service", copy=True) 
    .run_commands(
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
        "pip install -r /etc/systemd/requirements.txt",  # ✅ `run_commands()` 내에서 실행
    )
    .pip_install("ollama")
    .run_function(pull)  # Ollama 서버 실행 및 모델 다운로드
     # ✅ 마지막에 실행
)

app = modal.App(name="ollama", image=image)

with image.imports():
    import ollama
    from langchain_ollama import ChatOllama  # ✅ 최신 패키지 변경
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import ChatPromptTemplate


@app.cls(gpu="a10g", scaledown_window=300)  # ✅ 변경됨
class Ollama:
    @enter()
    def load(self):
        """Ollama 서버 실행"""
        subprocess.run(["systemctl", "start", "ollama"])

    @method(is_generator=True)  # ✅ Generator 사용
    def chain_ollama(self, topic: str):
        """
        주어진 주제에 대해 간략한 설명을 생성하는 함수.
        """
        llm = ChatOllama(model="gemma:7b")
        prompt = ChatPromptTemplate.from_template("{topic}에 대하여 간략히 설명해 줘.")
        chain = prompt | llm | StrOutputParser()
        yield from chain.stream({"topic": topic})  # ✅ Generator 반환


@app.local_entrypoint()
def main(topic: str = "서울", model: str = "gemma:7b", lookup: bool = False):
    """Ollama 모델을 사용하여 텍스트 추론 실행"""
    if lookup:
        ollama = modal.Cls.lookup("ollama", "Ollama")
    else:
        ollama = Ollama()

    if topic:
        # `chain_ollama` 실행
        for chunk in ollama.chain_ollama.remote_gen(topic):  # ✅ Generator 사용
            print(chunk, end="", flush=True)
