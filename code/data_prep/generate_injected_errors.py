import pandas as pd
import numpy as np

# ========== 설정 ==========
TRAIN_PATH = "data/processed/train70.csv"
TEST_PATH = "data/processed/test30.csv"
ERROR_OUT_CSV = "data/processed/generated_error.csv"
TRAIN_OUT = "data/processed/train_ready_with_error.csv"
TEST_OUT = "data/processed/test_ready_with_error.csv"
NUM_ERROR_TRAIN = 4000
NUM_ERROR_TEST = 1000

MEANINGFUL_COLUMNS = [
    'mqtt.kalive', 'mqtt.qos', 'mqtt.retain', 'mqtt.len', 'mqtt.dupflag',
    'mqtt.proto_len', 'mqtt.msgtype', 'mqtt.protoname'
]

# ========== 이상치 생성 함수 ==========
def inject_error(row):
    error_columns = np.random.choice(MEANINGFUL_COLUMNS, size=np.random.randint(2, 4), replace=False)
    for col in error_columns:
        if col == 'mqtt.kalive':
            row[col] = np.random.choice([-1, 0, 9999])
        elif col == 'mqtt.qos':
            row[col] = np.random.choice([5, 10, 999])
        elif col == 'mqtt.retain':
            row[col] = np.random.choice([-10, 50, 100])
        elif col == 'mqtt.len':
            row[col] = np.random.choice([0, -1, 10000])
        elif col == 'mqtt.dupflag':
            row[col] = np.random.choice([2, 999])
        elif col == 'mqtt.proto_len':
            row[col] = np.random.choice([-5, 0, 1000])
        elif col == 'mqtt.msgtype':
            row[col] = np.random.choice([-1, 50, 99])
        elif col == 'mqtt.protoname':
            row[col] = 'CORRUPTED'
    row['target'] = 'error'
    return row

# ========== train 처리 ==========
print("🚀 train70.csv 로드 중...")
train_chunks = pd.read_csv(TRAIN_PATH, chunksize=10000, low_memory=False)
filtered_train = []
for chunk in train_chunks:
    legit = chunk[chunk['target'] == 'legitimate']
    non_zero = ~(legit['tcp.flags'].notna() & (legit[MEANINGFUL_COLUMNS].fillna(0) == 0).all(axis=1))
    filtered = legit[non_zero]
    filtered_train.append(filtered)

train_legit_df = pd.concat(filtered_train, ignore_index=True)
train_sample = train_legit_df.sample(n=NUM_ERROR_TRAIN, random_state=42).reset_index(drop=True)
error_train = train_sample.apply(inject_error, axis=1)

# ========== test 처리 ==========
print("🚀 test30.csv 로드 중...")
test_df = pd.read_csv(TEST_PATH, low_memory=False)
error_test = error_train.sample(n=NUM_ERROR_TEST, random_state=42).reset_index(drop=True)

# ========== 삽입 위치 계산 ==========
def insert_errors(base_df, errors, interval):
    base_copy = base_df.copy().reset_index(drop=True)
    insert_idx = list(range(0, len(base_df), interval))[:len(errors)]

    # 삽입 위치에 index 열 추가
    errors = errors.copy()
    errors['__insert_idx__'] = insert_idx

    # base에도 index 열 추가
    base_copy['__insert_idx__'] = np.arange(len(base_copy))

    # concat 후 정렬
    combined = pd.concat([base_copy, errors], ignore_index=True)
    combined.sort_values(by='__insert_idx__', inplace=True)
    combined.drop(columns='__insert_idx__', inplace=True)

    return combined.reset_index(drop=True)


# 삽입
train_ready = insert_errors(pd.read_csv(TRAIN_PATH, low_memory=False), error_train, len(pd.read_csv(TRAIN_PATH, low_memory=False)) // NUM_ERROR_TRAIN)
test_ready = insert_errors(test_df, error_test, len(test_df) // NUM_ERROR_TEST)

# 저장
print(f"💾 저장 중: {TRAIN_OUT}")
train_ready.to_csv(TRAIN_OUT, index=False)
print(f"💾 저장 중: {TEST_OUT}")
test_ready.to_csv(TEST_OUT, index=False)
print(f"💾 이상치 샘플 저장: {ERROR_OUT_CSV}")
error_train.to_csv(ERROR_OUT_CSV, index=False)
print("✅ 완료.")
