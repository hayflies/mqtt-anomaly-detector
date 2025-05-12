import pandas as pd
import numpy as np
import joblib
from config import FEATURES
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 & 인코더 불러오기
xgb_model_3class = joblib.load("mqtt_xgb_classifier/model/xgb_model_3class.pkl")
xgb_model_maltype = joblib.load("mqtt_xgb_classifier/model/xgb_model_maltype.pkl")
label1_encoder = joblib.load("mqtt_xgb_classifier/model/label1_encoder.pkl")
label2_encoder = joblib.load("mqtt_xgb_classifier/model/label2_encoder.pkl")
tcp_flags_encoder = joblib.load("mqtt_xgb_classifier/model/tcp_flags_encoder.pkl")
conack_flags_encoder = joblib.load("mqtt_xgb_classifier/model/conack_flags_encoder.pkl")
conflags_encoder = joblib.load("mqtt_xgb_classifier/model/conflags_encoder.pkl")
hdrflags_encoder = joblib.load("mqtt_xgb_classifier/model/hdrflags_encoder.pkl")
msg_encoder = joblib.load("mqtt_xgb_classifier/model/msg_encoder.pkl")
protoname_encoder = joblib.load("mqtt_xgb_classifier/model/protoname_encoder.pkl")

def safe_transform(encoder, series):
    known_classes = set(encoder.classes_)
    series = series.astype(str)
    series = series.apply(lambda x: x if x in known_classes else "__unknown__")
    if "__unknown__" not in encoder.classes_:
        encoder.classes_ = np.append(encoder.classes_, "__unknown__")
    return encoder.transform(series)

# target → label1, label2 변환 함수
def convert_target_to_labels(df):
    df[['label1', 'label2']] = df['target'].apply(lambda x: pd.Series(
        ('normal', 'none') if x == 'legitimate' else
        ('ERROR', None) if x == 'error' else
        ('malicious', x)
    ))
    return df

def encode_message_columns(df):
    df['tcp.flags'] = safe_transform(tcp_flags_encoder, df['tcp.flags'])
    df['mqtt.conack.flags'] = safe_transform(conack_flags_encoder, df['mqtt.conack.flags'])
    df['mqtt.conflags'] = safe_transform(conflags_encoder, df['mqtt.conflags'])
    df['mqtt.hdrflags'] = safe_transform(hdrflags_encoder, df['mqtt.hdrflags'])
    df['mqtt.msg'] = safe_transform(msg_encoder, df['mqtt.msg'])
    df['mqtt.protoname'] = safe_transform(protoname_encoder, df['mqtt.protoname'])
    return df

def evaluate_model(test_path, model1, model2):
    df = pd.read_csv(test_path, usecols=FEATURES + ['target'], low_memory=False).dropna()
    df = convert_target_to_labels(df)
    df = encode_message_columns(df)

    df['label1'] = label1_encoder.transform(df['label1'])
    y1_pred = model1.predict(df[FEATURES])
    print("[1] 삼분류 평가:\n", classification_report(df['label1'], y1_pred, target_names=label1_encoder.classes_))

    mal_idx = y1_pred == label1_encoder.transform(['malicious'])[0]
    if mal_idx.any():
        y2_true = safe_transform(label2_encoder, df.loc[mal_idx, 'label2'])
        y2_pred = model2.predict(df.loc[mal_idx, FEATURES])
        print("[2] 악성코드 종류 평가:\n", classification_report(y2_true, y2_pred, target_names=label2_encoder.classes_))
        plot_confusion(y2_true, y2_pred, label2_encoder.classes_, "Malware Type Confusion Matrix")

def generate_combined_predictions(test_path, model1, model2):
    df = pd.read_csv(test_path, usecols=FEATURES + ['target'], low_memory=False).dropna()
    df = convert_target_to_labels(df)
    df = encode_message_columns(df)

    y1_pred = model1.predict(df[FEATURES])
    label1_str = label1_encoder.inverse_transform(y1_pred)
    final_preds = []
    for i, lbl in enumerate(label1_str):
        if i % 10000 == 0:
            print(f"🔁 {i}번째 샘플 예측 중...")
        if lbl == "malicious":
            mal_type_pred = model2.predict(df.loc[[i], FEATURES])
            mal_type_str = label2_encoder.inverse_transform(mal_type_pred)[0]
            final_preds.append(f"malicious_{mal_type_str}")
        else:
            final_preds.append(lbl)
    return final_preds

def save_predictions(pred_list, out_path):
    pd.DataFrame({"final_label": pred_list}).to_csv(out_path, index=False)
    print(f"예측 결과 저장 완료: {out_path}")

def plot_confusion(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Malgun Gothic'
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()
