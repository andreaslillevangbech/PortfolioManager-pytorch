config = {
    "layers":
    [
        {"filter_shape": [1, 3], "filter_number": 2, "weight_decay": 0.0},
        {"filter_number":20, "regularizer": "L2", "weight_decay": 5e-9},
        {"type": "EIIE_Output_WithW","regularizer": "L2", "weight_decay": 5e-8}
    ],
    'training':{
        'steps':5000,
        'batch_size': 50,
        'buffer_bias': 5e-5,
        'learning_rate': 0.00028,
        'fast_train': False,
        'decay_rate': 1.0,
        'decay_steps': 50000
    },
    "random_seed": 0,
    'input':{
        'global_period': 1800,
        'coin_no': 11,
        'window_size': 50,
        'feature_no': 3,
        'start_date': "2020/03/01",
        'end_date': "2020/07/01",
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
