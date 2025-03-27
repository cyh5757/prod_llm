import pandas as pd
import os

# 1. 데이터 불러오기
def load_raw_data(filepath="data/drug_raw.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {filepath}")
    return pd.read_csv(filepath)

# 2. 전처리 및 Chunk 생성 함수
def create_chunks(df):
    df.fillna("", inplace=True)  # 결측값 처리

    def make_chunk(row):
        return f"""
        약품명: {row['itemName']}
        제조사: {row['entpName']}
        효능: {row['efcyQesitm']}
        복용 방법: {row['useMethodQesitm']}
        주의사항: {row['atpnWarnQesitm']} {row['atpnQesitm']}
        상호작용: {row['intrcQesitm']}
        부작용: {row['seQesitm']}
        보관 방법: {row['depositMethodQesitm']}
        """.strip()

    df['chunk'] = df.apply(make_chunk, axis=1)
    return df[['itemName', 'chunk']]

# 3. 저장 함수
def save_chunks(df, filename="data/drug_chunks.csv"):
    df.to_csv(filename, index=False)
    print(f"💾 chunk 저장 완료: {filename}")

# 4. CLI 실행
if __name__ == "__main__":
    print("🧪 Step 2: 약품 데이터 전처리 및 Chunk 생성 중...")
    try:
        df_raw = load_raw_data()
        df_chunks = create_chunks(df_raw)
        save_chunks(df_chunks)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
