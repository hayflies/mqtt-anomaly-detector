import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import os

# 📥 1. 데이터 불러오기
df = pd.read_csv("data/processed/cleaned_binary.csv")

# 🎯 2. 특성과 라벨 정의
features = [
    'tcp.flags', 'tcp.time_delta',
    'mqtt.kalive', 'mqtt.qos', 'mqtt.retain', 'mqtt.len',
    'mqtt.dupflag', 'mqtt.msgtype'
]
X = df[features]
y = df['binary_label']

# 🧪 3. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔄 4. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⚖️ 5. scale_pos_weight 계산 (공격 비율 적을수록 더 높게 설정됨)
normal_count = sum(y_train == 0)
attack_count = sum(y_train == 1)
scale_pos_weight = (normal_count / attack_count) * 0.5

# 🚀 6. 모델 학습
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.5,  # L1 규제
    reg_lambda=1.0, # L2 규제
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 📊 7. 평가
y_pred = model.predict(X_test_scaled)

recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("📊 이진 분류 성능 리포트:\n", classification_report(y_test, y_pred))
print(f"🎯 공격(1) 클래스 기준 Recall: {recall:.4f}")
print(f"🎯 공격(1) 클래스 기준 F1-Score: {f1:.4f}")

# 💾 8. 모델 저장
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_binary_model.pkl")
joblib.dump(scaler, "models/xgb_binary_scaler.pkl")
print("✅ 모델과 스케일러 저장 완료 (models/xgb_binary_model.pkl)")
