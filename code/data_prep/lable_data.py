import pandas as pd

# 📥 1. CSV 불러오기
df = pd.read_csv("data/processed/raw_data.csv", low_memory=False)

# ✅ 2. 사용할 컬럼만 추리기
useful_cols = [
    'tcp.flags', 'tcp.time_delta',
    'mqtt.kalive', 'mqtt.qos', 'mqtt.retain', 'mqtt.len',
    'mqtt.dupflag', 'mqtt.msgtype',
    'attack_type'
]
df = df[useful_cols].copy()

# ✅ 3. tcp.flags: 16진수 문자열 → 10진수 정수 변환
df['tcp.flags'] = df['tcp.flags'].apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else x)

# ✅ 4. NaN 채우기 (mqtt 관련 필드들)
df.fillna({
    'mqtt.kalive': 0,
    'mqtt.qos': 0,
    'mqtt.retain': 0,
    'mqtt.len': 0,
    'mqtt.dupflag': 0,
    'mqtt.msgtype': -1,
}, inplace=True)

# ✅ 5. 이진 라벨 생성
df['binary_label'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)

# ✅ 6. 저장
df.to_csv("data/processed/cleaned_binary.csv", index=False)
print("✅ 전처리 완료: tcp.flags 변환 + 결측값 처리 + 이진 라벨 생성")