config = {
    'training':{
        'steps':100,
        'batch_size': 16,
        'buffer_bias': 5e-5
    },
    'input':{
        'global_period': 1800,
        'coin_no': 11,
        'window_size': 50,
        'feature_no': 3,
        'start_date': "2020/06/01",
        'end_date': "2020/07/09",
        "test_portion": 0.08,
        "volume_average_days": 30,
        "market": "poloniex",
        "online": 1
    },
    'trading':{
    "trading_consumption": 0.0025
    },
}


import os

DATABASE_DIR = os.getcwd() + "/" + "Data.db"
API_KEY = "ZOABM1Q7-9IPFLGUU-SPDARGFL-PZVP8U8X"
SECRET = "e42c1b3c920a32ff1b57765c12b225248cbf57889ee39cc95b4a5af09cbaa6cfcb9a1028ff89a1c1096ef09380a77758757714f6df5e222b03b8a1dd113a08ca"
REF_COIN = "BTC"