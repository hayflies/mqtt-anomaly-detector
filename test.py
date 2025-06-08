import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 불러오기
file_path = 'data/processed/train70_oversampled.csv'  # 🔁 여기에 실제 파일 경로 입력
df = pd.read_csv(file_path)

# 기본 정보 출력
print(df.info())
print(df.describe())

# 시각화: 숫자형 컬럼의 히스토그램
df.hist(figsize=(16, 12), bins=30)
plt.suptitle("📊 Numeric Column Distributions")
plt.tight_layout()
plt.show()

# 시각화: 범주형 컬럼의 분포 (상위 N개만 표시)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index[:10])
    plt.title(f"🧮 {col} Value Counts (Top 10)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
