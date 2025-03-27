# 1_fetch_data.py

import requests
import time
import xml.etree.ElementTree as ET
import pandas as pd
from urllib.parse import quote
from dotenv import load_dotenv
import os

# === 1. 환경 변수 로딩 ===
load_dotenv()
SERVICE_KEY = os.getenv("DRUG_DATA_API_KEY")

if SERVICE_KEY is None:
    raise ValueError("❌ .env 파일에 'DRUG_DATA_API_KEY'가 정의되어 있지 않습니다.")

# === 2. 기본 설정 ===
URL = 'http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList'
ROWS_PER_PAGE = 100


# === 3. 기능 함수 정의 ===
def get_total_count() -> int:
    params = {'serviceKey': SERVICE_KEY, 'pageNo': '1', 'numOfRows': '1', 'type': 'xml'}
    response = requests.get(URL, params=params)
    tree = ET.fromstring(response.content)
    return int(tree.find('body/totalCount').text)


def fetch_all_drug_data() -> pd.DataFrame:
    total_count = get_total_count()
    rows = []
    page = 1
    while True:
        params = {
            'serviceKey': SERVICE_KEY,
            'pageNo': str(page),
            'numOfRows': str(ROWS_PER_PAGE),
            'type': 'xml'
        }
        response = requests.get(URL, params=params)
        tree = ET.fromstring(response.content)
        items = tree.findall('body/items/item')
        if not items:
            break
        for item in items:
            row = {
                'itemName': item.findtext('itemName'),
                'entpName': item.findtext('entpName'),
                'efcyQesitm': item.findtext('efcyQesitm'),
                'useMethodQesitm': item.findtext('useMethodQesitm'),
                'atpnQesitm': item.findtext('atpnQesitm'),
                'seQesitm': item.findtext('seQesitm'),
            }
            rows.append(row)
        print(f"✅ {page} 페이지 수집 완료")
        page += 1
        time.sleep(0.2)
        if len(rows) >= total_count:
            break
    return pd.DataFrame(rows)


def save_to_csv(df: pd.DataFrame, filename: str = "drug_raw.csv"):
    df.to_csv(filename, index=False)
    print(f"📁 데이터 저장 완료 → {filename}")


# === 4. CLI 실행용 ===
if __name__ == "__main__":
    print("🚀 공공 API 약품 정보 수집 시작...")
    df = fetch_all_drug_data()
    save_to_csv(df)
