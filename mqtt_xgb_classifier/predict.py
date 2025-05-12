from config import *
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def predict_pipeline():
    # ====== 모델 & 인코더 로드 ======
    df = pd.read_csv(TEST_PATH, low_memory=False).fillna(0)
    model1 = joblib.load(MODEL1_PATH)
    model2 = joblib.load(MODEL2_PATH)
    le1 = joblib.load(ENC1_PATH)
    le2 = joblib.load(ENC2_PATH)

    # ====== label1 생성 ======
    def convert_target(x):
        if x == 'legitimate':
            return 'normal'
        elif x == 'error':
            return 'ERROR'
        else:
            return 'malicious'

    df['label1'] = df['target'].apply(convert_target)

    # ====== model1 예측 ======
    X = df[FEATURES]
    y1_pred = model1.predict(X)
    y1_true = le1.transform(df['label1'])
    label1_str = le1.inverse_transform(y1_pred)

    print("\n[1] ✅ 삼분류 평가 결과:")
    print(classification_report(y1_true, y1_pred, target_names=le1.classes_))

    # ====== confusion matrix (3-class) ======
    cm1 = confusion_matrix(y1_true, y1_pred)
    plt.figure(figsize=(6,5))
    plt.rcParams['font.family'] = 'Malgun Gothic'
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=le1.classes_, yticklabels=le1.classes_)
    plt.title("3-Class Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ====== model2 예측 (malicious만) ======
    final_pred = []
    y2_true, y2_pred = [], []

    print("\n[2] ✅ 악성코드 공격 타입 예측 중...")
    # model2 예측 (malicious만)
    for i, l1 in enumerate(label1_str):
        if i % 10000 == 0:
            print(f"🔁 {i}번째 샘플 예측 중...")

        if l1 == 'malicious':
            x_row = X.iloc[[i]]
            pred2 = model2.predict(x_row)[0]
            label2 = le2.inverse_transform([pred2])[0]
            final_pred.append(f"malicious_{label2}")

            # ✅ 오직 malicious인 경우에만 target을 인코딩
            if df.loc[i, 'target'] not in ['legitimate', 'error']:
                y2_true.append(le2.transform([df.loc[i, 'target']])[0])
                y2_pred.append(pred2)
        else:
            final_pred.append(l1)

    # ====== 악성코드 세부 분류 평가 ======
    if y2_true:
        print("\n[3] ✅ 악성코드 공격 타입 평가 결과:")
        print(classification_report(y2_true, y2_pred, target_names=le2.classes_))

        cm2 = confusion_matrix(y2_true, y2_pred)
        plt.figure(figsize=(8,6))
        plt.rcParams['font.family'] = 'Malgun Gothic'
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges', xticklabels=le2.classes_, yticklabels=le2.classes_)
        plt.title("Malware Type Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ 악성 예측 결과가 없어 세부 분류 평가 생략됨.")

    # ====== 저장 (유의미한 컬럼만 포함) ======
    df['final_label'] = final_pred
    df[USEFUL_COLS].to_csv(SAVE_PATH, index=False)
    print("📁 예측 결과 저장 완료 (유의미한 컬럼만):", SAVE_PATH)

