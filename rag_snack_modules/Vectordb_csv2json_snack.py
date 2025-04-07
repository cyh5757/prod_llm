# CSV 파일들을 로드
import pandas as pd
import ast
import json

# CSV 파일 경로
snack_df = pd.read_csv("raw_snack_data/snack.csv")
snack_item_df = pd.read_csv("raw_snack_data/snack_item.csv")
snack_additive_df = pd.read_csv("raw_snack_data/snack_additive.csv")
map_df = pd.read_csv("raw_snack_data/map_snack_item_additive.csv")

# RAG용 JSON 구조 생성
rag_documents = []

for i, item in snack_item_df.iterrows():
    snack = snack_df[snack_df['id'] == item['snack_id']]
    if snack.empty:
        continue
    snack = snack.iloc[0]

    additive_ids = map_df[map_df['snack_item_id'] == item['id']]['snack_additive_id'].tolist()
    additives = snack_additive_df[snack_additive_df['id'].isin(additive_ids)]

    additive_texts = []
    for _, row in additives.iterrows():
        try:
            uses = ast.literal_eval(row['main_use_list']) if pd.notnull(row['main_use_list']) else []
        except:
            uses = []
        additive_texts.append(
            f"- {row['korean_name']} ({row['grade']} 등급) / 용도: {', '.join(uses)} / 설명: {row['description']}"
        )

    try:
        nutrients = json.loads(item['nutrient_list']) if pd.notnull(item['nutrient_list']) else []
    except:
        nutrients = []

    nutrient_texts = [
        f"{n['nutrient']}: {n['servingAmountInfo']['amount']}{n['servingAmountInfo']['amountUnit']}"
        for n in nutrients
    ]

    page_content = "\n".join([
        f"간식명: {snack['name']}",
        f"제조사: {snack['company']}",
        f"종류: {snack['snack_type']}",
        f"제공단위: {item['service_unit']}",
        f"열량: {item['calorie']} kcal",
        f"총 제공량: {snack['total_serving_size']}",
        "",
        "📌 영양 정보:",
        " / ".join(nutrient_texts),
        "",
        "📌 첨가물:",
        "\n".join(additive_texts),
        "",
        f"📌 알레르기 정보: {', '.join(json.loads(snack['allergy_list'])) if pd.notnull(snack['allergy_list']) else '없음'}",
        f"📌 인증 마크: {', '.join(json.loads(snack['safe_food_mark_list'])) if pd.notnull(snack['safe_food_mark_list']) else '없음'}"
    ])

    rag_documents.append({
        "page_content": page_content,
        "metadata": {
            "filename": f"snack_{i}"
        }
    })

# JSON 파일로 저장
output_path = "distilated_snack_data/Vectordb_formatted_snack_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(rag_documents, f, ensure_ascii=False, indent=2)

output_path

