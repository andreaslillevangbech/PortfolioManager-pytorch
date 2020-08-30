from datetime import datetime
import time, calendar
import pandas as pd

from src.constants import *
from src.data.poloniex import Poloniex

start = int(time.mktime(start.timetuple()) - time.timezone)
end= int(calendar.timegm(end.timetuple()))

tick = polo.marketTicker()
tick['USDC_BTC']

time_index = pd.to_datetime(list(range(start, end+1, FIFTEEN_MINUTES)),unit='s')
print(time_index)
print(len(time_index))