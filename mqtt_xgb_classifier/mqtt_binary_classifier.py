import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score
from xgboost import XGBClassifier
import joblib
import os

# 📥 데이터 불러오기
df = pd.read_csv("data/processed/cleaned_binary.csv")

# 🎯 특성 및 라벨
features = [
    'tcp.flags', 'tcp.time_delta',
    'mqtt.kalive', 'mqtt.qos', 'mqtt.retain', 'mqtt.len',
    'mqtt.dupflag', 'mqtt.msgtype'
]
X = df[features]
y = df['binary_label']

# 🧪 train/test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🔄 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ⚖️ class imbalance 비율 계산
normal_count = sum(y_train == 0)
attack_count = sum(y_train == 1)
scale_pos_weight = normal_count / attack_count

# 🚀 모델 학습
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 🎯 예측 확률로부터 threshold 적용
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# 🔥 threshold 설정 (0.6 ~ 0.7 정도로 실험 가능)
threshold = 0.70
y_pred = (y_probs >= threshold).astype(int)

# 📊 평가
print(f"✅ Threshold = {threshold}")
print("📊 정상 분류 리포트:\n", classification_report(y_test, y_pred))
print("\n\n\t정상\n\t공격")

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"🎯 Recall(공격): {recall:.4f}")
print(f"🎯 Precision(공격): {precision:.4f}")
print(f"🎯 F1-score(공격): {f1:.4f}")

# 💾 모델 저장
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_binary_model.pkl")
joblib.dump(scaler, "models/xgb_binary_scaler.pkl")
print("✅ 모델과 스케일러 저장 완료")