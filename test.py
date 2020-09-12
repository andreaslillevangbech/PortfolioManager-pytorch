from datetime import datetime
import time, calendar
import pandas as pd

from src.constants import *
from src.data.poloniex import Poloniex
from src.data.coinlist import CoinList

polo = Poloniex()


start = datetime(2020, 1, 20, 3, 0)
end= datetime(2020, 1, 20, 4, 20)

start = int(time.mktime(start.timetuple()) - time.timezone)
end= int(calendar.timegm(end.timetuple()))
period = FIFTEEN_MINUTES

coin = CoinList(end = end)
print(coin.topNVolume(n=10))

