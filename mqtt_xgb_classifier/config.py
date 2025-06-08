FEATURES = ['mqtt.kalive', 'mqtt.qos', 'mqtt.msgtype', 'tcp.len']
USEFUL_COLS = [
    'tcp.flags', 'tcp.time_delta',
    'mqtt.kalive', 'mqtt.qos', 'mqtt.retain', 'mqtt.len', 'mqtt.dupflag',
    'mqtt.proto_len', 'mqtt.msgtype', 'mqtt.protoname', 'final_label'
]

TRAIN_PATH = "data/processed/train70_oversampled.csv"
TEST_PATH = "data/processed/test30_balanced.csv"
SAVE_PATH = "mqtt_xgb_classifier/output/final_predictions.csv"

MODEL_PATH = "mqtt_xgb_classifier/model/lightGBM_model.pkl"