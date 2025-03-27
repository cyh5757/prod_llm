# 4_rag_agent.py
# 목적: RAG 기반 약품 정보 검색 에이전트

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import LangChainTracer
from langchain.schema import Document
from typing import List
import pandas as pd

# 1. 환경변수 로드
load_dotenv()

# 2. LangSmith 추적 설정
from langchain_teddynote import logging
logging.langsmith("5_rag_agent")

# 3. Pinecone 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-db")
NAMESPACE = "drug-rag-namespace"

# 4. 임베딩 모델 및 벡터 스토어 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)

# 5. LLM 모델 초기화
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.7
)

# 6. 프롬프트 템플릿 정의
prompt_template = """다음은 약품 정보에 대한 질문과 답변 형식입니다:

질문: {question}

답변: 다음 약품 정보를 참고하여 답변해주세요:
{context}

답변 형식:
1. 약품명: [약품명]
2. 효능/효과: [효능/효과]
3. 사용법: [사용법]
4. 주의사항: [주의사항]
5. 부작용: [부작용]
6. 상호작용: [상호작용]

답변:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 7. RAG 체인 구성
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 3}
    ),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 8. Gradio 인터페이스 정의
def query_drug_info(query: str) -> str:
    """약품 정보를 검색하고 결과를 반환합니다."""
    try:
        result = chain({"query": query})
        answer = result["result"]
        sources = result["source_documents"]
        
        # 소스 문서 정보 추가
        source_info = "\n\n참고한 약품 정보:\n"
        for i, doc in enumerate(sources, 1):
            source_info += f"{i}. {doc.metadata.get('itemName', '알 수 없음')}\n"
        
        return answer + source_info
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

# 9. Gradio 인터페이스 생성
iface = gr.Interface(
    fn=query_drug_info,
    inputs=gr.Textbox(
        lines=2,
        placeholder="약품에 대해 궁금한 점을 입력하세요...",
        label="질문"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="답변"
    ),
    title="💊 약품 정보 검색 시스템",
    description="약품의 효능, 사용법, 주의사항 등을 검색할 수 있습니다.",
    examples=[
        "아스피린의 효능과 주의사항이 궁금합니다.",
        "감기약 복용 시 주의할 점을 알려주세요.",
        "혈압약과 함께 먹으면 안 되는 약이 있나요?"
    ]
)

# 10. 서버 실행
if __name__ == "__main__":
    print("🚀 약품 정보 검색 시스템 시작...")
    iface.launch(share=True) 