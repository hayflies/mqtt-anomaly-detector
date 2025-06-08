import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
import os

# 📥 1. 데이터 불러오기
df = pd.read_csv("data/processed/cleaned_binary.csv")

# 🎯 2. 공격 데이터만 필터링
df_attack = df[df['binary_label'] == 1].copy()

# 📌 3. 다중 클래스 레이블 인코딩 (ex: flood → 0, bruteforce → 1 ...)
label_encoder = LabelEncoder()
df_attack['multi_label'] = label_encoder.fit_transform(df_attack['attack_type'])

# 💾 레이블 맵 저장 (나중에 역변환용)
os.makedirs("models", exist_ok=True)
joblib.dump(label_encoder, "models/attack_label_encoder.pkl")

# 🎯 4. 특성과 라벨 정의
features = [
    'tcp.flags', 'tcp.time_delta',
    'mqtt.kalive', 'mqtt.qos', 'mqtt.retain', 'mqtt.len',
    'mqtt.dupflag', 'mqtt.msgtype'
]
X = df_attack[features]
y = df_attack['multi_label']

# 🧪 5. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔄 6. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🚀 7. 모델 학습
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softmax',  # 다중 클래스 분류
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 📊 8. 평가
y_pred = model.predict(X_test_scaled)
print("👾 공격 타입 분류 리포트:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 💾 9. 모델 & 스케일러 저장
joblib.dump(model, "models/xgb_attack_type_model.pkl")
joblib.dump(scaler, "models/xgb_attack_type_multiclass_scaler.pkl")
print("✅ 공격 분류 모델 및 인코더 저장 완료")
