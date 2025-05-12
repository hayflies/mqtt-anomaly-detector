import pandas as pd
from sqlalchemy import create_engine
import os

# ▶️ MySQL 연결 정보 (수정 필요)
db_user = 'mqtt_database'       # 예: root
db_password = 'software507'   # 예: 1234
db_host = 'localhost'           # 또는 127.0.0.1
db_port = '3306'
db_name = 'mqtt_database'
table_name = 'mqtt_data_union'

# ▶️ 의미 있는 컬럼 리스트
useful_columns = [
    'tcp.flags', 'tcp.time_delta', 'tcp.len', 'mqtt.conack.flags', 'mqtt.conack.val',
    'mqtt.conflag.cleansess', 'mqtt.conflag.passwd', 'mqtt.conflag.uname', 'mqtt.conflags',
    'mqtt.dupflag', 'mqtt.hdrflags', 'mqtt.kalive', 'mqtt.len', 'mqtt.msg', 'mqtt.msgid',
    'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.qos', 'mqtt.retain', 'mqtt.ver'
]

# ▶️ 파일과 라벨 매핑
file_label_map = {
    'data/raw/bruteforce.csv': 'bruteforce',
    'data/raw/flood.csv': 'flood',
    'data/raw/malaria.csv': 'malaria',
    'data/raw/malformed.csv': 'malformed',
    'data/raw/slowite.csv': 'slowite',
    'data/raw/legitimate_1w.csv': 'legitimate'
}

# ▶️ raw 파일 읽고 통합
df_list = []

for file_path, label in file_label_map.items():
    if os.path.exists(file_path):
        print(f"📂 {file_path} 로딩 중...")
        df = pd.read_csv(file_path, low_memory=False)
        df = df[useful_columns].copy()
        df['target'] = label
        df_list.append(df)
        print(f"✅ {label} 데이터 {len(df)}행 로드 완료")
    else:
        print(f"⚠️ 파일 없음: {file_path}")

# ▶️ 전체 데이터프레임 통합
print("🔄 전체 데이터프레임 병합 중...")
full_df = pd.concat(df_list, ignore_index=True)
print(f"📊 전체 데이터 수: {len(full_df)}")

# ▶️ SQLAlchemy로 DB 연결 및 적재
print("🔌 DB 연결 중...")
connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

print("📤 DB로 데이터 적재 중... (1000행씩 나눠서 처리)")
full_df.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=1000)

print("✅ 모든 데이터가 MySQL에 성공적으로 적재되었습니다.")