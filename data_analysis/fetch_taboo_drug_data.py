import requests
import time
import pandas as pd
from dotenv import load_dotenv
import os
import xml.etree.ElementTree as ET
import urllib3


# 1. 환경 변수 로드
load_dotenv()
SERVICE_KEY = os.getenv("DRUG_DATA_API_DECODED_KEY")

if SERVICE_KEY is None:
    raise ValueError("❌ .env 파일에 DRUG_DATA_API_DECODED_KEY 항목이 없습니다.")

# 2. API 기본 설정
BASE_URL = "http://apis.data.go.kr/1471000/DURPrdlstInfoService03/getUsjntTabooInfoList03"
ROWS_PER_PAGE = 100

# 3. 성분명으로 병용금기 정보 조회
def fetch_taboo_by_drfstf(item_name):
    rows = []
    page = 1
    while True:
        params = {
            'serviceKey': SERVICE_KEY,
            'pageNo': str(page),
            'numOfRows': str(ROWS_PER_PAGE),
            'type': 'xml',
            'itemName': item_name
        }
        try:
            response = requests.get(BASE_URL, params=params, verify=False, timeout=10)
            if response.status_code != 200:
                print(f"❌ [{item_name}] 페이지 {page} 요청 실패: {response.status_code}")
                break

            root = ET.fromstring(response.content)
            items = root.findall('.//item')

            if not items:
                print(f"⚠️ [{item_name}] 페이지 {page} 결과 없음.")
                break

            for item in items:
                rows.append({
                    '조회성분명': item_name,
                    '품목명': item.findtext('ITEM_NAME'),
                    '성분명': item.findtext('INGR_NAME'),
                    '금기약품명': item.findtext('PROHBT_ITEM_NAME'),
                    '금기내용': item.findtext('PROHBT_CONTENT'),
                    '혼합금기대상약품명': item.findtext('MIXTURE_ITEM_NAME')
                })

            print(f"✅ [{item_name}] 페이지 {page} 완료 (누적 {len(rows)}건)")
            page += 1
            time.sleep(0.1)

        except Exception as e:
            print(f"❌ [{item_name}] 오류 발생: {e}")
            break

    return rows

# 4. 전체 성분명 기반 병용금기 정보 수집
def fetch_all_from_narcotic_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'DRFSTF' not in df.columns:
        raise ValueError("❌ 'DRFSTF' 컬럼이 존재하지 않습니다.")

    unique_items = df['DRFSTF'].dropna().unique()
    all_rows = []

    for item_name in unique_items:
        print(f"\n🚀 병용금기 수집 시작: {item_name}")
        result = fetch_taboo_by_drfstf(item_name)
        all_rows.extend(result)
        time.sleep(0.2)

    return pd.DataFrame(all_rows)

# 5. 결과 저장 함수
def save_to_csv(df, filename="data/taboo_from_drfstf.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.drop_duplicates(inplace=True)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n💾 저장 완료: {filename}")

# 6. 실행
if __name__ == "__main__":
    try:
        print("🚀 병용금기 정보 수집 시작 (from narcotic_drug_list.csv)...")
        df_result = fetch_all_from_narcotic_csv("data\\narcotic_drug_list.csv")
        if not df_result.empty:
            save_to_csv(df_result)
        else:
            print("⚠️ 수집된 데이터가 없습니다.")
    except Exception as e:
        print(f"❌ 전체 오류 발생: {e}")
