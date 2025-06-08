import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from config import TRAIN_PATH, TEST_PATH, MODEL_PATH
import warnings
warnings.filterwarnings('ignore')

# 🔹 클래스 정의
class_names = [ 'legitimate', 'dos', 'flood', 'bruteforce', 'malformed', 'slowite']
target_mapping = {name: idx for idx, name in enumerate(class_names)}
inverse_mapping = {idx: name for name, idx in target_mapping.items()}

# 🔹 데이터 로딩
df_train = pd.read_csv(TRAIN_PATH, low_memory=False)
df_test = pd.read_csv(TEST_PATH, low_memory=False)

# 🔹 타겟 인코딩
df_train['target'] = df_train['target'].map(target_mapping)
df_test['target'] = df_test['target'].map(target_mapping)

# 🔹 피처와 타겟 분리
X_train = df_train.drop('target', axis=1)
y_train = df_train['target']
X_test = df_test.drop('target', axis=1)
y_test = df_test['target']

# 🔹 문자열 → 코드화
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = X_train[col].astype('category').cat.codes
        X_test[col] = X_test[col].astype('category').cat.codes

# 🔹 클래스 가중치 → 샘플 가중치 적용
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
weights = np.array([class_weights[label] for label in y_train])

# 🔹 DMatrix 생성
dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
dvalid = xgb.DMatrix(X_test, label=y_test)

alpha_dict = {
    0: 0.25,  # legitimate (줄임)
    1: 0.25,  # dos (줄임)
    2: 0.5,  # flood
    3: 0.5,  # bruteforce
    4: 0.5,  # malformed
    5: 0.25  # slowite (줄임)
}

# 🔹 Focal Loss 정의
def focal_loss(alpha_dict, gamma=2.0, num_classes=6):
    def focal_obj(preds, dtrain):
        labels = dtrain.get_label().astype(int)
        preds = preds.reshape(-1, num_classes)
        preds = np.clip(preds, -10, 10)
        exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        one_hot = np.eye(num_classes)[labels]
        pt = (probs * one_hot).sum(axis=1).reshape(-1, 1)
        alpha_vec = np.array([alpha_dict[label] for label in labels]).reshape(-1, 1)
        grad = -alpha_vec * (1 - pt) ** gamma * (one_hot - probs)
        hess = alpha_vec * (1 - pt) ** gamma * (1 - probs) * probs * (gamma + 1)
        return grad.reshape(-1), hess.reshape(-1)
    return focal_obj

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

print("🚀 XGBoost + Focal Loss 학습 시작...")
evals = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    obj=focal_loss(alpha_dict, gamma=2.0, num_classes=len(class_names)),
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

# 🔹 분류 리포트 출력
print("✅ 분류 리포트:")
report_dict = classification_report(
    y_test, y_pred_classes,
    target_names=class_names,
    output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
print(report_df)

# 🔹 시각화 - 성능 지표
report_df = report_df.iloc[:-3, :]
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
plt.title("Per-Class Precision, Recall, F1 Score")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True)
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
