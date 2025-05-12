import pandas as pd
from sklearn.utils import resample

# 기존 데이터 로드
df = pd.read_csv("data/processed/train_ready_balanced.csv")

# error 샘플만 추출
error_df = df[df['target'] == 'error']
non_error_df = df[df['target'] != 'error']

# error 샘플 3배 증강 (1000 → 3000)
error_upsampled = resample(error_df, replace=True, n_samples=3000, random_state=42)

# 병합 + 셔플
df_balanced = pd.concat([non_error_df, error_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

# 저장
df_balanced.to_csv("data/processed/train_ready_balanced_errorboosted.csv", index=False)
print("✅ 증강된 train 파일 저장 완료: train_ready_balanced_errorboosted.csv")