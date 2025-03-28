# import pandas as pd
# import json

# # CSV 파일 로드
# snack_df = pd.read_csv("raw_snack_data/snack.csv")
# snack_item_df = pd.read_csv("raw_snack_data/snack_item.csv")
# snack_additive_df = pd.read_csv("raw_snack_data/snack_additive.csv")
# map_df = pd.read_csv("raw_snack_data/map_snack_item_additive.csv")

# # 데이터 병합 및 JSON 변환
# formatted_records = []

# for _, item in snack_item_df.iterrows():
#     snack = snack_df[snack_df['id'] == item['snack_id']].iloc[0]
#     additive_ids = map_df[map_df['snack_item_id'] == item['id']]['snack_additive_id'].tolist()
#     additives = snack_additive_df[snack_additive_df['id'].isin(additive_ids)]

#     additive_list = []
#     for _, row in additives.iterrows():
#         uses = json.loads(row['main_use_list']) if pd.notnull(row['main_use_list']) else []
#         additive_info = {
#             "korean_name": row['korean_name'],
#             "grade": row['grade'],
#             "uses": uses,
#             "description": row['description'],
#             "stability_message": row['stability_message'] if pd.notnull(row['stability_message']) else '정보 없음'
#         }
#         additive_list.append(additive_info)

#     nutrients = json.loads(item['nutrient_list']) if pd.notnull(item['nutrient_list']) else []
#     nutrient_list = [
#         {
#             "nutrient": n['nutrient'],
#             "amount": n['servingAmountInfo']['amount'],
#             "unit": n['servingAmountInfo']['amountUnit']
#         } for n in nutrients
#     ]

#     record = {
#         "snack_name": snack['name'],
#         "manufacturer": snack['company'],
#         "type": snack['snack_type'],
#         "serving_unit": item['service_unit'],
#         "calories": item['calorie'],
#         "total_serving_size": snack['total_serving_size'],
#         "nutrients": nutrient_list,
#         "additives": additive_list,
#         "allergy_info": json.loads(snack['allergy_list']) if pd.notnull(snack['allergy_list']) else [],
#         "certifications": json.loads(snack['safe_food_mark_list']) if pd.notnull(snack['safe_food_mark_list']) else []
#     }
#     formatted_records.append(record)

# # JSON 파일로 저장
# with open("distilled_snack_data/formatted_snack_data.json", "w", encoding="utf-8") as f:
#     json.dump(formatted_records, f, ensure_ascii=False, indent=4)
