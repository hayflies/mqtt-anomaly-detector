import pandas as pd
import xgboost as xgb
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from config import FEATURES, TRAIN_PATH, MODEL1_PATH, MODEL2_PATH, ENC1_PATH, ENC2_PATH

# ========= 라벨 생성 함수 ==========
def convert_labels(df):
    df['label1'] = df['target'].apply(lambda x:
        'normal' if x == 'legitimate'
        else 'ERROR' if x == 'error'
        else 'malicious')
    df['label2'] = df['target'].apply(lambda x:
        'none' if x == 'legitimate'
        else None if x == 'error'
        else x)
    return df

# ========= 모델 및 인코더 저장 함수 ==========
def save_models(model1, model2, le1, le2):
    joblib.dump(model1, MODEL1_PATH)
    joblib.dump(model2, MODEL2_PATH)
    joblib.dump(le1, ENC1_PATH)
    joblib.dump(le2, ENC2_PATH)
    print("✅ 모델1|모델2, 인코더 저장 완료.")

# ========= 전체 학습 함수 ==========
def train_models(CHUNKSIZE=100000):
    # 진행률 표시하며 CSV 로딩
    print("📥 데이터 로딩 중...")
    total_rows = sum(1 for _ in open(TRAIN_PATH)) - 1  # 헤더 제외
    chunks = []
    for chunk in tqdm(pd.read_csv(TRAIN_PATH, chunksize=CHUNKSIZE, low_memory=False),
                      desc="Loading CSV", total=total_rows // CHUNKSIZE + 1):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True).fillna(0)

    print("✅ CSV 전체 로딩 완료. 총 행 수:", len(df))

    # 라벨 생성
    df = convert_labels(df)

    # 인코딩
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    y1 = le1.fit_transform(df['label1'])

    # 모델1 학습: 3-class 분류
    print("🧠 모델1 (3-class) 학습 중...")
    model1 = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        max_depth=6,
        learning_rate=0.07,
        n_estimators=200
    )
    model1.fit(df[FEATURES], y1)

    # 모델2 학습: 악성 데이터의 세부 유형 분류
    print("🧠 모델2 (malicious subtype) 학습 중...")
    mal_df = df[df['label1'] == 'malicious'].copy()
    y2 = le2.fit_transform(mal_df['label2'])
    model2 = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le2.classes_),
        eval_metric='mlogloss',
        max_depth=6,
        learning_rate=0.07,
        n_estimators=200
    )
    model2.fit(mal_df[FEATURES], y2)

    # 모델 및 인코더 저장
    save_models(model1, model2, le1, le2)