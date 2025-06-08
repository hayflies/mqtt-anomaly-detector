import pandas as pd
import os

# 📁 CSV 경로
base_dir = "data/raw"

# 🗂️ 파일별 공격 타입 정의
label_map = {
    "legitimate_1w.csv": "normal",
    "bruteforce.csv": "bruteforce",
    "flood.csv": "flood",
    "malaria.csv": "dos",
    "malformed.csv": "malformed",
    "slowite.csv": "slowite"
}

# 🧪 DataFrame 병합
df_list = []
for file, attack_type in label_map.items():
    path = os.path.join(base_dir, file)
    temp_df = pd.read_csv(path)
    temp_df["attack_type"] = attack_type  # 라벨 추가
    df_list.append(temp_df)

# 🔗 통합
df_total = pd.concat(df_list, ignore_index=True)

# ✅ 저장
save_path = "data/processed/raw_data.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
df_total.to_csv(save_path, index=False)
print(f"📁 모든 CSV를 병합해 저장 완료: {save_path}")
