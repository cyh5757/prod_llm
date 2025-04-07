import os
from pprint import pprint
from dotenv import load_dotenv
import logging

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.extractor import KeyphraseExtractor
from ragas.testset.docstore import InMemoryDocumentStore

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    try:
        # ✅ 환경변수 불러오기 (API KEY 등)
        load_dotenv()

        # ✅ 문서 로딩
        loader = JSONLoader(
            file_path="distilated_snack_data/Vectordb_formatted_snack_data.json",
            jq_schema=".[]",         # JSON 배열의 각 항목을 개별 문서로 로드
            text_content=False       # page_content 필드가 이미 존재함
        )
        docs = loader.load()
        logger.info(f"✅ 총 {len(docs)}개 문서 로드됨")
        logger.debug("📄 첫 문서 내용 일부:\n%s", docs[0].page_content[:300])

        # ✅ 문서 메타데이터 정리
        for doc in docs:
            doc.metadata["filename"] = doc.metadata.get("source", "unknown")

        # ✅ LLM 및 임베딩 설정
        generator_llm = ChatOpenAI(model="gpt-4")  # 모델명 수정
        critic_llm = ChatOpenAI(model="gpt-4")     # 모델명 수정
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # ✅ 래퍼 및 도구 구성
        langchain_llm = LangchainLLMWrapper(generator_llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
        keyphrase_extractor = KeyphraseExtractor(llm=langchain_llm)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        # ✅ 문서 저장소 초기화
        docstore = InMemoryDocumentStore(
            splitter=splitter,
            embeddings=ragas_embeddings,
            extractor=keyphrase_extractor,
        )

        # ✅ 테스트셋 생성기 구성
        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            ragas_embeddings,
            docstore=docstore,
        )

        # ✅ 질문 유형 비율 설정
        distributions = {
            simple: 0.4,
            reasoning: 0.2,
            multi_context: 0.2,
            conditional: 0.2,
        }

        # ✅ 테스트셋 생성
        testset = generator.generate_with_langchain_docs(
            documents=docs[:5],  # 더 작은 샘플로 테스트
            test_size=3,         # 더 작은 테스트 크기
            distributions=distributions,
            with_debugging_logs=True,
            raise_exceptions=False,  # 예외 발생 시 경고만 표시
        )

        # ✅ 결과 저장
        df = testset.to_pandas()
        df.to_csv("distilated_snack_data/ragas_synthetic_dataset.csv", index=False)
        logger.info("✅ 테스트셋 CSV 저장 완료")

    except Exception as e:
        logger.error("오류 발생: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
