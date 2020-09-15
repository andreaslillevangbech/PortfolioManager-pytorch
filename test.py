#!/usr/bin env python
from datetime import datetime
import time, calendar
import pandas as pd
import time

from src.constants import *
from src.data.poloniex import Poloniex
from src.data.coinlist import CoinList

start = datetime(2020, 8, 20, 3, 0)
end= datetime(2020, 8, 20, 4, 20)

start = int(time.mktime(start.timetuple()) - time.timezone)
end= int(calendar.timegm(end.timetuple()))
period = FIFTEEN_MINUTES
now = int(time.time())

coin = CoinList(end = now)
print(coin.topNVolume(n=15))

