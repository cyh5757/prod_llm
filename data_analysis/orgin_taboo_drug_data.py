import requests
import time
import pandas as pd
from dotenv import load_dotenv
import os
import xml.etree.ElementTree as ET

# 1. 환경 변수 로드
load_dotenv()
SERVICE_KEY = os.getenv("DRUG_DATA_API_DECODED_KEY")

if SERVICE_KEY is None:
    raise ValueError("❌ .env 파일에 DRUG_DATA_API_DECODED_KEY 항목이 없습니다.")

# 2. API 기본 설정
BASE_URL = "http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03"
ROWS_PER_PAGE = 100

# 3. 전체 데이터 수집
def fetch_all_taboo_drug_data():
    rows = []
    page = 1

    while True:
        params = {
            'serviceKey': SERVICE_KEY,
            'pageNo': str(page),
            'numOfRows': str(ROWS_PER_PAGE),
            'type': 'xml'
        }
        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            print(f"❌ 페이지 {page} 요청 실패: {response.status_code}")
            break

        root = ET.fromstring(response.content)
        items = root.findall('.//item')

        if not items:
            print(f"⚠️ {page} 페이지 항목 없음. 종료")
            break

        for item in items:
            rows.append({
                'ITEM_NAME': item.findtext('ITEM_NAME'),
                'INGR_NAME': item.findtext('INGR_NAME'),
                'PROHBT_ITEM_NAME': item.findtext('PROHBT_ITEM_NAME'),
                'PROHBT_CONTENT': item.findtext('PROHBT_CONTENT'),
                'MIXTURE_ITEM_NAME': item.findtext('MIXTURE_ITEM_NAME')
            })

        print(f"✅ {page} 페이지 수집 완료 ({len(rows)} 누적)")
        page += 1
        time.sleep(0.1)

    return pd.DataFrame(rows)

# 4. 마약 관련 키워드 필터링
def filter_narcotic_related(df):
    keyword = "마약"
    filtered = df[
        df.apply(lambda row: keyword in str(row['ITEM_NAME']) + str(row['INGR_NAME']) + str(row['PROHBT_CONTENT']), axis=1)
    ]
    print(f"🔍 마약 관련 병용금기 항목 수: {len(filtered)}")
    return filtered

# 5. 저장 함수
def save_to_csv(df, filename="data/narcotic_taboo_interactions.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"💾 저장 완료: {filename}")

# 6. 실행
if __name__ == "__main__":
    print("🚀 병용금기 데이터 수집 시작...")
    try:
        df_all = fetch_all_taboo_drug_data()
        df_filtered = filter_narcotic_related(df_all)
        save_to_csv(df_filtered)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
