import pandas as pd
from sklearn.utils import shuffle

# 기존 파일 불러오기
train = pd.read_csv("data/processed/train_ultimate_v2.csv", low_memory=False)
test = pd.read_csv("data/processed/test_ultimate_v2.csv", low_memory=False)
error_for_train = pd.read_csv("data/raw/error_synthetic_diverse_4000_v2.csv", low_memory=False)  # target=error 포함
error_for_test = pd.read_csv("data/raw/error_synthetic_diverse_4000_v2.csv", low_memory=False)

# 병합
train_combined = pd.concat([train, error_for_train], ignore_index=True)
test_combined = pd.concat([test, error_for_test], ignore_index=True)

# 셔플
train_shuffled = shuffle(train_combined, random_state=42).reset_index(drop=True)
test_shuffled = shuffle(test_combined, random_state=42).reset_index(drop=True)

# 저장
train_shuffled.to_csv("data/processed/train_ultimate_v3.csv", index=False)
print("✅ 저장 완료: train_ultimate_v3.csv")
test_shuffled.to_csv("data/processed/test_ultimate_v3.csv", index=False)
print("✅ 저장 완료: test_ultimate_v3.csv")