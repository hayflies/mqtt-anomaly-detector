from train import train_models
from config import MODEL1_PATH, MODEL2_PATH
from predict import predict_pipeline
import joblib

# 1. 학습
print("🚀 모델 학습 시작...")
train_models()

# 2. 저장된 모델 로드
print("📥 저장된 모델 불러오는 중...")
xgb_model_3class = joblib.load(MODEL1_PATH)
xgb_model_maltype = joblib.load(MODEL2_PATH)

# 3. 예측, 평가 및 저장
print("🔎 예측, 평가 및 저장 파이프라인 시작...")
predict_pipeline()

print("✅ 전체 파이프라인 완료.")