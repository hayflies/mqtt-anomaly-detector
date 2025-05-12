FEATURES = [
    'tcp.flags', 'tcp.len', 'mqtt.conack.flags', 'mqtt.conack.val',
    'mqtt.conflag.cleansess', 'mqtt.conflag.passwd', 'mqtt.conflag.uname', 'mqtt.conflags',
    'mqtt.dupflag', 'mqtt.hdrflags', 'mqtt.kalive', 'mqtt.len', 'mqtt.msg', 'mqtt.msgid',
    'mqtt.msgtype', 'mqtt.proto_len', 'mqtt.protoname', 'mqtt.qos', 'mqtt.retain', 'mqtt.ver'
]

TRAIN_PATH = "data/processed/train_ultimate_v3.csv"
TEST_PATH = "data/processed/test_ultimate_v3.csv"
SAVE_PATH = "mqtt_xgb_classifier/output/final_predictions.csv"