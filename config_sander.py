#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import enum

class EndPoints(enum.Enum):
    returnTicker = 0
    return24Volume = 1
    returnOrderBook = 2
    returnTradeHistory = 3
    returnChartData = 4
    returnCurrencies = 5
    returnLoanOrders = 6

DATABASE_DIR = os.getcwd() + "/" + "Data.db"
API_KEY = "ZOABM1Q7-9IPFLGUU-SPDARGFL-PZVP8U8X"
SECRET = "e42c1b3c920a32ff1b57765c12b225248cbf57889ee39cc95b4a5af09cbaa6cfcb9a1028ff89a1c1096ef09380a77758757714f6df5e222b03b8a1dd113a08ca"
REF_COIN = "BTC"

CONFIG_FILE_DIR = 'net_config.json'
LAMBDA = 1e-4  # lambda in loss function 5 in training
   # About time

   
NOW = 0
FIVE_MINUTES = 60 * 5
FIFTEEN_MINUTES = FIVE_MINUTES * 3
HALF_HOUR = FIFTEEN_MINUTES * 2
HOUR = HALF_HOUR * 2
TWO_HOUR = HOUR * 2
FOUR_HOUR = HOUR * 4
DAY = HOUR * 24
YEAR = DAY * 365
   # trading table name
TABLE_NAME = 'test'