from train import train_model_in_chunks, save_models
import xgboost as xgb
from config import TRAIN_PATH, TEST_PATH, SAVE_PATH

# 모델 정의 (3-class, malware-type)
xgb_model_3class = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss')
xgb_model_maltype = xgb.XGBClassifier(objective='multi:softmax', num_class=5, eval_metric='mlogloss')

# 학습
print("🚀 학습 시작...")
train_model_in_chunks(TRAIN_PATH, xgb_model_3class, xgb_model_maltype)

# 모델 저장
print("💾 모델 저장 중...")
save_models(xgb_model_3class, xgb_model_maltype)

# 🔄 predict 관련 함수는 이 시점 이후 import
from predict import evaluate_model, generate_combined_predictions, save_predictions

# 평가
print("🔍 테스트 평가 중...")
evaluate_model(TEST_PATH, xgb_model_3class, xgb_model_maltype)

# 예측 라벨 생성 및 저장
print("📤 예측 결과 저장 중...")
final_preds = generate_combined_predictions(TEST_PATH, xgb_model_3class, xgb_model_maltype)
save_predictions(final_preds, SAVE_PATH)

print("✅ 전체 파이프라인 완료.")