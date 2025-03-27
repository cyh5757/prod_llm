import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from pinecone import Pinecone, ServerlessSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote import logging
# 목적: 사용자의 질문에 대해 Pinecone에서 유사 문서 검색 후 LLM 응답 생성

# LangSmith 추적 설정
logging.langsmith("4_query_rag_pinecone")

# 1. 환경변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-db")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
NAMESPACE = "drug-rag-namespace"

# 2. 모델 및 임베딩 초기화
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embedder = OpenAIEmbeddings(model="text-embedding-3-large")

# 3. Pinecone 인덱스 준비
def get_or_create_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
        print(f"✅ 인덱스 생성 완료: {PINECONE_INDEX_NAME}")
    return pc.Index(PINECONE_INDEX_NAME)

index = get_or_create_index()

# 4. 검색 함수 정의
def similarity_search(query, top_k=5):
    embedded_query = embedder.embed_query(query)
    result = index.query(vector=embedded_query, top_k=top_k, namespace=NAMESPACE, include_metadata=True)
    return [
        Document(page_content=match["metadata"].get("itemName", "") + "\n" + match["metadata"].get("text", ""))
        for match in result["matches"]
    ]

# 5. 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template("""
너는 의약품 정보를 설명해주는 전문가야. 아래의 약품 정보를 참고해서 사용자 질문에 친절하게 답변해줘.

약품 정보:
{context}

사용자 질문:
{question}
""")

# 6. 체인 구성
rag_chain = (
    RunnableMap({
        "context": lambda x: "\n\n".join([doc.page_content for doc in similarity_search(x["question"])]),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 7. CLI 실행
if __name__ == "__main__":
    print("💬 약품 질문 시스템 (Pinecone + LLM)")
    try:
        while True:
            query = input("🔍 질문을 입력하세요 (종료: 'exit'): ")
            if query.lower() in ["exit", "quit"]:
                break
            response = rag_chain.invoke({"question": query})
            print(f"🧠 응답: {response}\n")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")