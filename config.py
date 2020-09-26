config = {
    'training':{
        'steps':100000,
        'batch_size': 109,
        'buffer_bias': 5e-5,
        'fast_train': False
    },
    'input':{
        'global_period': 1800,
        'coin_no': 11,
        'window_size': 31,
        'feature_no': 3,
        'start_date': "2019/12/01",
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
REF_COIN = "BTC"
