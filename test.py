from datetime import datetime
import time
import pandas as pd

from src.constants import *
from src.data.poloniex import Poloniex

minute = 60
five_minutes = 5*minute

polo = Poloniex()

start = datetime(2020, 1, 20, 3, 9)
start = int(time.mktime(start.timetuple()))

end= datetime(2020, 1, 20, 3, 20)
end= int(time.mktime(end.timetuple()))


chart = polo.marketChart(period=five_minutes, start=start, end=end, pair = 'USDC_BTC')
for i in chart:
    print(i)

print(len(chart))

first = chart[0]['date']
print(pd.to_datetime(start))
print(pd.to_datetime(first))

last = chart[-1]['date']
print(pd.to_datetime(end))
print(pd.to_datetime(last))
