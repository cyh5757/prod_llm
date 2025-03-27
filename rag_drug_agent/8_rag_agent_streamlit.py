# 목적: Streamlit UI로 약품 RAG 에이전트 제공 + 피드백 저장

import os
import streamlit as st
import csv
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_teddynote import logging
logging.langsmith("8_rag_agent_streamlit")

# 환경변수 로드
load_dotenv()

# 설정 값 로딩
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-db")
NAMESPACE = "drug-rag-namespace"

# 임베딩 및 벡터스토어 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)

# LLM
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)

# 프롬프트 템플릿
template = """
다음은 약품 정보에 대한 질문과 답변 형식입니다:

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

답변:
"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# RAG 체인
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 질문-응답-출처 기록 저장
LOG_PATH = "logs/streamlit_feedback_log.csv"
os.makedirs("logs", exist_ok=True)

def save_feedback_log(query, answer, sources, feedback):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_names = ", ".join([doc.metadata.get("itemName", "N/A") for doc in sources])
    with open(LOG_PATH, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, query, answer, source_names, feedback])

# Streamlit UI
st.set_page_config(page_title="💊 약품 RAG 챗봇", layout="centered")
st.title("💊 약품 정보 검색 에이전트")
st.markdown("GPT-4 + Pinecone 기반으로 약품 정보를 검색해드립니다.")

query = st.text_input("궁금한 점을 입력하세요:", placeholder="예: 타이레놀 부작용은?")

if query:
    with st.spinner("답변 생성 중..."):
        result = chain.invoke({"query": query})
        answer = result["result"]
        sources = result["source_documents"]

        source_list = "\n".join([f"- {doc.metadata.get('itemName', '알 수 없음')}" for doc in sources])
        st.markdown("#### 📌 답변")
        st.markdown(answer)
        st.markdown("#### 📚 참고한 약품 정보")
        st.code(source_list, language="")

        # 피드백 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 도움이 되었어요"):
                save_feedback_log(query, answer, sources, "positive")
                st.success("피드백 감사합니다!")
        with col2:
            if st.button("👎 부족했어요"):
                save_feedback_log(query, answer, sources, "negative")
                st.warning("더 나은 답변을 위해 노력할게요!")