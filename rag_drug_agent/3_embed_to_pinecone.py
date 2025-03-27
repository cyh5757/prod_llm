# 3_embed_to_pinecone.py (rag_drug_agent용)
# 목적: drug_chunks.csv를 기반으로 Pinecone에 벡터 업로드

import os
import pandas as pd
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict, Any
# LangSmith 추적 설정
from langchain_teddynote import logging
logging.langsmith("3_embed_to_pinecone")
# 1. 환경변수 로드
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-db")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
NAMESPACE = "drug-rag-namespace"

# 2. 데이터 로드 및 Document 변환
def load_documents(filepath="data/drug_chunks.csv"):
    df = pd.read_csv(filepath)
    df.fillna("", inplace=True)
    documents = [
        Document(
            page_content=row["chunk"],
            metadata={"itemName": row["itemName"]}
        )
        for _, row in df.iterrows()
    ]
    return documents

# 3. Pinecone 인덱스 생성 또는 연결
def get_or_create_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=3072,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
        )
        print(f"✅ Pinecone 인덱스 생성 완료: {PINECONE_INDEX_NAME}")

    return pc.Index(PINECONE_INDEX_NAME)

# 4. 배치 처리 함수
def process_batch(batch: List[Document], embeddings: OpenAIEmbeddings, index: Any):
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]

    vectors = embeddings.embed_documents(texts)

    index.upsert(
        vectors=[
            (f"doc_{i}", vector, metadata)
            for i, (vector, metadata) in enumerate(zip(vectors, metadatas))
        ],
        namespace=NAMESPACE
    )

# 5. 메인 실행
if __name__ == "__main__":
    print("🚀 약품 정보 벡터 저장 시작...")

    try:
        print("📄 문서 로드 중...")
        documents = load_documents()

        print("📌 Pinecone 인덱스 준비 중...")
        index = get_or_create_index()

        print("🔗 임베딩 모델 준비 중...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        BATCH_SIZE = 64
        batches = [documents[i:i + BATCH_SIZE] for i in range(0, len(documents), BATCH_SIZE)]

        print("🚀 벡터 업로드 중...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_batch, batch, embeddings, index)
                for batch in batches
            ]

            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                pass

        print("✅ 벡터 저장 완료: Pinecone")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")