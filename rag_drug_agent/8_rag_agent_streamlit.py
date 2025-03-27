import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 0. 초기 설정 및 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-db")
NAMESPACE = "drug-rag-namespace"

# 1. 벡터스토어 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. LLM 및 프롬프트 세팅 (질문에 맞는 정보만 추출하게 유도)
template = """
너는 약학 전문 상담 챗봇이야. 아래의 약품 정보를 참고하여 사용자의 질문에 **관련된 내용만** 골라서 간결하게 답변해줘.

약품 정보:
{context}

사용자 질문:
{question}

💬 답변:
- 사용자의 질문과 관련된 약품 정보만 요약해서 알려줘.
- 질문과 관련 없는 정보는 출력하지 마.
"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 3. Streamlit UI
st.set_page_config(page_title="💊 약품 검색 비교", layout="centered")
st.title("🔍 RAG vs 키워드 검색 비교")

mode = st.sidebar.radio("검색 모드 선택", ["RAG 응답 (GPT 포함)", "키워드 기반 문서 검색"])
query = st.text_input("질문을 입력하세요", placeholder="예: 타이레놀의 부작용은?")

if query:
    with st.spinner("검색 중..."):
        if mode == "RAG 응답 (GPT 포함)":
            result = rag_chain.invoke({"query": query})
            st.subheader("📌 GPT 응답")
            st.markdown(result["result"])

            st.subheader("📚 참고 문서")
            for doc in result["source_documents"]:
                st.markdown(f"**{doc.metadata.get('itemName', '알 수 없음')}**")
                st.code(doc.page_content.strip()[:1000])

        else:
            docs = retriever.invoke(query)
            st.subheader("📄 유사 문서 결과")
            for i, doc in enumerate(docs, 1):
                st.markdown(f"### {i}. {doc.metadata.get('itemName', '알 수 없음')}")
                st.code(doc.page_content.strip()[:1000])
