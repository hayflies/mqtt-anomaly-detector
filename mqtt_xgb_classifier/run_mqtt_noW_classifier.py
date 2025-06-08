import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import TRAIN_PATH, TEST_PATH, MODEL_PATH

# 🔹 클래스 정의
class_names = ['legitimate', 'dos', 'flood', 'bruteforce', 'malformed', 'slowite']
target_mapping = {name: idx for idx, name in enumerate(class_names)}
inverse_mapping = {idx: name for name, idx in target_mapping.items()}

# 🔹 데이터 로딩
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

# 🔹 타겟 인코딩
df_train['target'] = df_train['target'].map(target_mapping)
df_test['target'] = df_test['target'].map(target_mapping)

# 🔹 피처와 타겟 분리
X_train = df_train.drop('target', axis=1)
y_train = df_train['target']
X_test = df_test.drop('target', axis=1)
y_test = df_test['target']

# 🔹 문자열 타입 컬럼 수치형으로 변환
def convert_object_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df

X_train = convert_object_to_numeric(X_train)
X_test = convert_object_to_numeric(X_test)

# 🔹 DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

# 🔹 XGBoost 하이퍼파라미터
params = {
    'objective': 'multi:softprob',
    'num_class': len(class_names),
    'eval_metric': 'mlogloss',
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'lambda': 1,
    'alpha': 0.1,
    'seed': 42
}

# 🔹 학습
print("🚀 XGBoost 학습 시작...")
evals = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=100
)

# 🔹 모델 저장
model.save_model(MODEL_PATH)
print(f"💾 모델 저장 완료: {MODEL_PATH}")

# 🔹 예측
print("🔍 테스트셋 예측 중...")
y_pred = model.predict(dvalid)
y_pred_classes = np.argmax(y_pred, axis=1)

# 🔹 리포트 출력
print("✅ 분류 리포트:")
report_dict = classification_report(
    y_test, y_pred_classes,
    target_names=class_names,
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
print(report_df)

# 🔹 시각화 - 클래스별 성능
report_df = report_df.iloc[:-3, :]
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
plt.title("Per-Class Precision, Recall, F1 Score")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# 🔹 Feature Importance 시각화
print("📊 Feature Importance 시각화")
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=15, importance_type='gain', ax=ax)
plt.title("Top 15 Feature Importance (by Gain)")
plt.tight_layout()
plt.show()

# 🔹 혼동 행렬
cm = confusion_matrix(y_test, y_pred_classes, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Reds', fmt='.2f', xticklabels=class_names, yticklabels=class_names)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
