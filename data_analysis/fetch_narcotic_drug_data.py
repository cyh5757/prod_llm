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
BASE_URL = "http://apis.data.go.kr/1471000/NrcdGnrlzInfoService01/getNrcdGnrlzList"
ROWS_PER_PAGE = 100

# 3. 전체 마약류 목록 수집
def fetch_all_narcotic_drugs():
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
                'DRUG_NO': item.findtext('DRUG_NO'),
                'DRFSTF': item.findtext('DRFSTF'),           # 마약성 약물 이름 (예: 펜타닐)
                'DRFSTF_ENG': item.findtext('DRFSTF_ENG'),   # 영문명
                'PHARM': item.findtext('PHARM'),             # 약효군
                'TYPE_CODE': item.findtext('TYPE_CODE'),     # 구분 코드 (예: 마약)
                'SIDE_EFFECT': item.findtext('SIDE_EFFECT'), # 부작용
                'MEDICATION': item.findtext('MEDICATION')    # 투여 방법
            })

        print(f"✅ {page} 페이지 수집 완료 ({len(rows)} 누적)")
        page += 1
        time.sleep(0.1)

    return pd.DataFrame(rows)

# 4. 저장 함수
def save_to_csv(df, filename="data/narcotic_drug_list.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.drop_duplicates(subset=["DRUG_NO"], inplace=True)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"💾 저장 완료: {filename}")

# 5. 실행
if __name__ == "__main__":
    print("🚀 마약성 약물 목록 수집 시작...")
    try:
        df = fetch_all_narcotic_drugs()
        save_to_csv(df)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
