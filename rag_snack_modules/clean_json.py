import json

file_path = "distilated_snack_data/formatted_snack_data_vectordb.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 비정상 항목 확인 및 처리
cleaned_data = []
for i, entry in enumerate(data):
    content = entry.get("page_content")
    if isinstance(content, str):
        cleaned_data.append(entry)
    elif isinstance(content, dict) or isinstance(content, list):
        # dict나 list인 경우 문자열로 덤프
        entry["page_content"] = json.dumps(content, ensure_ascii=False)
        cleaned_data.append(entry)
    else:
        print(f"⚠️ {i}번째 항목이 비정상 page_content:", type(content))

# 저장
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"✅ 총 {len(cleaned_data)}개의 정상 항목 저장 완료.")