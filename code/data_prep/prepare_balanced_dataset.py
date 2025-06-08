import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def cleandata(df):
    for col in [
        'frame.time_invalid', 'frame.time_epoch', 'frame.time_relative', 'frame.number',
        'frame.time_delta', 'frame.time_delta_displayed', 'frame.cap_len', 'frame.len',
        'tcp.window_size_value', 'eth.src', 'eth.dst', 'ip.src', 'ip.dst', 'ip.proto',
        'tcp.srcport', 'tcp.dstport', 'tcp.analysis.initial_rtt', 'tcp.stream',
        'mqtt.topic', 'tcp.checksum', 'mqtt.topic_len', 'mqtt.passwd_len', 'mqtt.passwd',
        'mqtt.clientid', 'mqtt.clientid_len', 'mqtt.username', 'mqtt.username_len'
    ]:
        if col in df.columns:
            del df[col]
    return df

# 🔹 CSV 읽기
print("🚀 원본 CSV 불러오는 중...")
df_legitimate = pd.read_csv('data/raw/legitimate_1w.csv').fillna(0)
df_legitimate['target'] = 'legitimate'

df_slowite = pd.read_csv('data/raw/slowite.csv').fillna(0)
df_slowite['target'] = 'slowite'

df_malaria = pd.read_csv('data/raw/malaria.csv').fillna(0)
df_malaria['target'] = 'dos'

df_malformed = pd.read_csv('data/raw/malformed.csv').fillna(0)
df_malformed['target'] = 'malformed'

df_flood = pd.read_csv('data/raw/flood.csv').fillna(0)
df_flood['target'] = 'flood'

df_bruteforce = pd.read_csv('data/raw/bruteforce.csv').fillna(0)
df_bruteforce['target'] = 'bruteforce'

# 🔹 legitimate 샘플링 (예: 2%)
print("📉 legitimate 샘플링 중...")
df_legitimate_sampled = df_legitimate.sample(frac=0.02, random_state=42)

# 🔹 병합
df = pd.concat([
    df_legitimate_sampled, df_slowite, df_malaria,
    df_malformed, df_flood, df_bruteforce
], ignore_index=True)

df = shuffle(df, random_state=42)
df = cleandata(df)

# 🔹 훈련/테스트셋 분할 (7:3)
print("🧪 훈련/테스트셋 분할 중...")
train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['target'], random_state=42)

# 🔹 저장
train_df.to_csv("data/processed/train70_balanced.csv", index=False)
test_df.to_csv("data/processed/test30_balanced.csv", index=False)

print("✅ 저장 완료: train70_oversampled.csv / test30_balanced.csv")
print(train_df['target'].value_counts())
print(test_df['target'].value_counts())
