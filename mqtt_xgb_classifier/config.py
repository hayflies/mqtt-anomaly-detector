FEATURES = ['mqtt.kalive', 'mqtt.qos', 'mqtt.msgtype', 'tcp.len']
USEFUL_COLS = [
    'tcp.flags', 'tcp.time_delta',
    'mqtt.kalive', 'mqtt.qos', 'mqtt.retain', 'mqtt.len', 'mqtt.dupflag',
    'mqtt.proto_len', 'mqtt.msgtype', 'mqtt.protoname', 'final_label'
]

TRAIN_PATH = "data/processed/train_ultimate.csv"
TEST_PATH = "data/processed/test_ultimate.csv"
SAVE_PATH = "mqtt_xgb_classifier/output/final_predictions.csv"

MODEL1_PATH = "mqtt_xgb_classifier/model/xgb_model_3class.pkl"
MODEL2_PATH = "mqtt_xgb_classifier/model/xgb_model_maltype.pkl"
ENC1_PATH = "mqtt_xgb_classifier/model/label1_encoder.pkl"
ENC2_PATH = "mqtt_xgb_classifier/model/label2_encoder.pkl"