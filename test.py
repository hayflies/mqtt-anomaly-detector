import pandas as pd
import matplotlib.pyplot as plt

# 파일 경로 지정 (train 또는 test)
df = pd.read_csv("data/processed/train_ready.csv", low_memory=False)

str_cols = df.select_dtypes(include=['object', 'string']).columns
print("문자열 컬럼:", str_cols)
