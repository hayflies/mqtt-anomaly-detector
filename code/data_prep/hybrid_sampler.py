import pandas as pd
from sklearn.utils import resample
import os

# 파일 경로
TRAIN_PATH = "data/processed/train_ready.csv"
TEST_PATH = "data/processed/test_ready.csv"
OUTPUT_DIR = "data/processed"

# 하이브리드 샘플링 함수
def hybrid_sampling(df, max_legit=100000, min_target_counts=None):
    if min_target_counts is None:
        min_target_counts = {
            "dos": 10000,
            "bruteforce": 4000,
            "malformed": 3000,
            "slowite": 3000,
            "flood": 1000,
            "error": 1000
        }

    dfs = []

    # legitimate 언더샘플링
    legit_df = df[df['target'] == 'legitimate'].sample(n=max_legit, random_state=42)
    dfs.append(legit_df)

    # 나머지 클래스 오버샘플링 또는 그대로
    for label, min_count in min_target_counts.items():
        class_df = df[df['target'] == label]
        if len(class_df) < min_count:
            class_df_upsampled = resample(class_df,
                                          replace=True,
                                          n_samples=min_count,
                                          random_state=42)
            dfs.append(class_df_upsampled)
        else:
            dfs.append(class_df)

    return pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

# 데이터 로드
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 샘플링 적용
train_balanced = hybrid_sampling(train_df)
test_balanced = hybrid_sampling(test_df, max_legit=50000)

# 저장
train_out = os.path.join(OUTPUT_DIR, "train_ready_balanced.csv")
test_out = os.path.join(OUTPUT_DIR, "test_ready_balanced.csv")
train_balanced.to_csv(train_out, index=False)
test_balanced.to_csv(test_out, index=False)

print(f"✅ Balanced train saved to: {train_out}")
print(f"✅ Balanced test saved to: {test_out}")
