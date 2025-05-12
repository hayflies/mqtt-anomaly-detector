import pandas as pd
from tqdm import tqdm
from config import FEATURES
import joblib
from sklearn.preprocessing import LabelEncoder

# 인코더 객체 전역
label1_encoder = LabelEncoder()
label2_encoder = LabelEncoder()

tcp_flags_encoder = LabelEncoder()
conack_flags_encoder = LabelEncoder()
conflags_encoder = LabelEncoder()
hdrflags_encoder = LabelEncoder()
msg_encoder = LabelEncoder()
protoname_encoder = LabelEncoder()

def convert_target_to_labels(df):
    """ target → label1, label2 변환 """
    df[['label1', 'label2']] = df['target'].apply(lambda x: pd.Series(
        ('normal', 'none') if x == 'legitimate' else
        ('ERROR', None) if x == 'error' else
        ('malicious', x)
    ))
    return df

def encode_message_columns(df):
    df['tcp.flags'] = tcp_flags_encoder.fit_transform(df['tcp.flags'].astype(str))
    df['mqtt.conack.flags'] = conack_flags_encoder.fit_transform(df['mqtt.conack.flags'].astype(str))
    df['mqtt.conflags'] = conflags_encoder.fit_transform(df['mqtt.conflags'].astype(str))
    df['mqtt.hdrflags'] = hdrflags_encoder.fit_transform(df['mqtt.hdrflags'].astype(str))
    df['mqtt.msg'] = msg_encoder.fit_transform(df['mqtt.msg'].astype(str))
    df['mqtt.protoname'] = protoname_encoder.fit_transform(df['mqtt.protoname'].astype(str))
    return df

def train_model_in_chunks(path, model1, model2, chunksize=1000):
    chunk_iter = pd.read_csv(path, usecols=FEATURES + ['target'], chunksize=chunksize, low_memory=False)

    for chunk in tqdm(chunk_iter, desc="Training in chunks"):
        chunk.dropna(subset=FEATURES + ['target'], inplace=True)

        # ✅ chunk 내부 shuffle
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)

        # ✅ label1/label2 생성
        chunk = convert_target_to_labels(chunk)
        chunk = encode_message_columns(chunk)

        # ✅ label 인코딩
        chunk['label1'] = label1_encoder.fit_transform(chunk['label1'])
        X = chunk[FEATURES]
        y1 = chunk['label1']
        model1.fit(X, y1)

        # ✅ malicious만 따로 분리해서 label2 학습
        if 'label2' in chunk.columns and 'malicious' in label1_encoder.classes_:
            malicious_label_num = label1_encoder.transform(['malicious'])[0]
            mal_chunk = chunk[chunk['label1'] == malicious_label_num].copy()
            if not mal_chunk.empty:
                mal_chunk['label2'] = label2_encoder.fit_transform(mal_chunk['label2'])
                y2 = mal_chunk['label2']
                model2.fit(mal_chunk[FEATURES], y2)

def save_models(model1, model2):
    joblib.dump(model1, "mqtt_xgb_classifier/model/xgb_model_3class.pkl")
    joblib.dump(model2, "mqtt_xgb_classifier/model/xgb_model_maltype.pkl")
    joblib.dump(label1_encoder, "mqtt_xgb_classifier/model/label1_encoder.pkl")
    joblib.dump(label2_encoder, "mqtt_xgb_classifier/model/label2_encoder.pkl")
    joblib.dump(tcp_flags_encoder, "mqtt_xgb_classifier/model/tcp_flags_encoder.pkl")
    joblib.dump(conack_flags_encoder, "mqtt_xgb_classifier/model/conack_flags_encoder.pkl")
    joblib.dump(conflags_encoder, "mqtt_xgb_classifier/model/conflags_encoder.pkl")
    joblib.dump(hdrflags_encoder, "mqtt_xgb_classifier/model/hdrflags_encoder.pkl")
    joblib.dump(msg_encoder, "mqtt_xgb_classifier/model/msg_encoder.pkl")
    joblib.dump(protoname_encoder, "mqtt_xgb_classifier/model/protoname_encoder.pkl")