# 9_feedback_dashboard.py (최종 수정본)
import pandas as pd
import streamlit as st
from datetime import datetime
from collections import Counter

st.set_page_config(page_title="📊 피드백 대시보드", layout="wide")
st.title("📊 사용자 피드백 분석 대시보드")

LOG_PATH = "logs/streamlit_feedback_log.csv"

# 1. 캐싱된 데이터 로더
@st.cache_data
def load_feedback_log(path):
    try:
        # 첫 줄을 컬럼으로 인식하지 않도록 명시
        df = pd.read_csv(path, header=None, names=["timestamp", "query", "answer", "sources", "feedback"])
    except Exception as e:
        st.error(f"❌ CSV 로딩 실패: {e}")
        st.stop()

    # 컬럼 타입 변환
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
    except Exception as e:
        st.error(f"❌ timestamp 변환 실패: {e}")
        st.stop()

    return df


# 2. 대시보드 시각화
try:
    df = load_feedback_log(LOG_PATH)

    # 날짜 슬라이더 or 안내
    min_date = df["date"].min()
    max_date = df["date"].max()

    if min_date == max_date:
        st.info(f"📅 분석 가능한 날짜: {min_date} (하루치만 존재합니다)")
        date_range = (min_date, max_date)
    else:
        date_range = st.slider(
            "📅 분석할 날짜 범위 선택",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )

    df_filtered = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]

    st.subheader("✅ 피드백 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("총 질문 수", len(df_filtered))
    col2.metric("👍 긍정 피드백", (df_filtered["feedback"] == "positive").sum())
    col3.metric("👎 부정 피드백", (df_filtered["feedback"] == "negative").sum())

    st.divider()

    # 날짜별 피드백 추이
    st.subheader("📈 날짜별 피드백 추이")
    chart_data = df_filtered.groupby(["date", "feedback"]).size().unstack().fillna(0)
    st.line_chart(chart_data)

    # 자주 언급된 약품 Top 10
    st.subheader("💊 자주 언급된 약품 (출처 기반 Top 10)")
    all_sources = ", ".join(df_filtered["sources"].dropna()).split(", ")
    source_counts = Counter(all_sources)
    common_sources = pd.DataFrame(source_counts.items(), columns=["약품명", "언급 횟수"]).sort_values(by="언급 횟수", ascending=False).head(10)
    st.bar_chart(common_sources.set_index("약품명"))

    # 질문 로그 테이블
    st.subheader("📄 질문 상세 로그")
    st.dataframe(df_filtered[["timestamp", "query", "feedback", "sources"]].sort_values(by="timestamp", ascending=False))

except FileNotFoundError:
    st.error(f"❌ 로그 파일이 존재하지 않습니다: {LOG_PATH}")
