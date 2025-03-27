import requests
import time
import pandas as pd
from dotenv import load_dotenv
import os
import xml.etree.ElementTree as ET

# 1. .env 파일에서 환경변수 로드
load_dotenv()
SERVICE_KEY = os.getenv("DRUG_DATA_API_DECODED_KEY")  # 반드시 인코딩된 키여야 함

if SERVICE_KEY is None:
    raise ValueError("❌ .env 파일에 DRUG_DATA_API_KEY 항목이 없습니다.")

# 2. API 기본 설정
BASE_URL = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
ROWS_PER_PAGE = 100  # 이 API는 최대 100건까지 허용

# 3. 전체 건수 조회 함수 (XML 기반 파싱)
def get_total_count():
    params = {
        'serviceKey': SERVICE_KEY,
        'pageNo': '1',
        'numOfRows': '1',
        'type': 'xml'
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        raise ConnectionError(f"❌ 요청 실패: {response.status_code}")

    root = ET.fromstring(response.content)
    total_count = root.findtext('.//totalCount')
    if total_count is None:
        raise ValueError("❌ totalCount를 찾을 수 없습니다. 인증키 확인 필요")

    return int(total_count)

# 4. 전체 데이터 수집 함수 (XML 기반)
def fetch_all_drug_data():
    total_count = get_total_count()
    print(f"📦 총 {total_count}건 수집 예정")
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
            print(f"⚠️ {page} 페이지에서 항목 없음. 종료")
            break

        for item in items:
            rows.append({
                'itemSeq': item.findtext('itemSeq'),
                'itemName': item.findtext('itemName'),
                'entpName': item.findtext('entpName'),
                'efcyQesitm': item.findtext('efcyQesitm'),
                'useMethodQesitm': item.findtext('useMethodQesitm'),
                'atpnWarnQesitm': item.findtext('atpnWarnQesitm'),
                'atpnQesitm': item.findtext('atpnQesitm'),
                'intrcQesitm': item.findtext('intrcQesitm'),
                'seQesitm': item.findtext('seQesitm'),
                'depositMethodQesitm': item.findtext('depositMethodQesitm'),
                'openDe': item.findtext('openDe'),
                'updateDe': item.findtext('updateDe'),
            })

        print(f"✅ {page} 페이지 수집 완료 ({len(rows)} 누적)")
        page += 1
        time.sleep(0.1)

        if len(rows) >= total_count:
            break

    return pd.DataFrame(rows)

# 5. CSV 저장 함수
def save_to_csv(df, filename="data/drug_raw.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.drop_duplicates(subset=["itemSeq"], inplace=True)
    df.to_csv(filename, index=False)
    print(f"💾 저장 완료: {filename}")

# 6. CLI 실행
if __name__ == "__main__":
    print("🚀 공공 API 약품 정보(XML) 수집 시작...")
    try:
        df = fetch_all_drug_data()
        save_to_csv(df)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")