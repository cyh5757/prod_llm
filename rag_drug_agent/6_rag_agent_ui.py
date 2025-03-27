import os
import gradio as gr
import csv
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_teddynote import logging
# 목적: 약품 정보를 검색하는 RAG 기반 Agent + Gradio UI + 기록 저장 기능 (Step 6)
# 1. 환경변수 로드 및 LangSmith 추적 설정
load_dotenv()
logging.langsmith("6_rag_agent_ui")

# 2. 환경변수 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-db")
NAMESPACE = "drug-rag-namespace"

# 3. 임베딩 모델 및 벡터스토어 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)

# 4. LLM 모델 설정
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.7
)

# 5. 프롬프트 템플릿
prompt_template = """
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

답변:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 6. RAG 체인 구성
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# 7. 질문/응답 저장 함수
def save_log(query, answer, sources, path="logs/drug_query_log.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_names = ", ".join([doc.metadata.get("itemName", "N/A") for doc in sources])
    with open(path, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, query, answer, source_names])

# 8. 질의 함수 정의
def query_drug_info(query: str) -> str:
    try:
        result = chain.invoke({"query": query})
        answer = result["result"]
        sources = result["source_documents"]

        source_info = "\n\n📚 참고한 약품 정보:\n"
        for i, doc in enumerate(sources, 1):
            source_info += f"{i}. {doc.metadata.get('itemName', '알 수 없음')}\n"

        full_response = answer + source_info
        save_log(query, answer, sources)
        return full_response
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

# 9. Gradio UI 정의
def build_ui():
    with gr.Blocks(title="약품 검색 에이전트") as demo:
        gr.Markdown("""# 💊 약품 정보 검색 에이전트
        GPT-4 + Pinecone 기반으로 약품 정보를 검색해드립니다.
        """)
        query = gr.Textbox(
            label="질문",
            placeholder="예: 아스피린 효능과 부작용 알려줘"
        )
        output = gr.Textbox(label="답변")

        query.submit(fn=query_drug_info, inputs=query, outputs=output)

    return demo

# 10. 실행
if __name__ == "__main__":
    print("🚀 약품 검색 에이전트 UI 실행 중...")
    ui = build_ui()
    ui.launch(share=True)
