import langchain
import ragas
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_loaders import JSONLoader
from pydantic import BaseModel
logging.langsmith("Snack_MakeData_Evaluations")
load_dotenv()




loader = JSONLoader(
    file_path="distilated_snack_data/formatted_snack_data.json",
    jq_schema='.',  # JSON 배열의 각 요소를 Document로
)

docs = loader.load()
docs = docs[3:-1]  # 목차/마지막 제외
print(f"문서 개수: {len(docs)}")
print(docs[0].page_content)