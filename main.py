from src.data.datamatrices import DataMatrices
from datetime import datetime
import json
import time

with open("./src/net_config.json") as file:
    config = json.load(file)
    print(config)
    start = time.mktime(datetime.strptime(config["input"]["start_date"], "%Y/%m/%d").timetuple())
    end = time.mktime(datetime.strptime(config["input"]["end_date"], "%Y/%m/%d").timetuple())
    DataMatrices(start=start,
                    end=end,
                    feature_number=config["input"]["feature_number"],
                    window_size=config["input"]["window_size"],
                    online=True,
                    period=config["input"]["global_period"],
                    volume_average_days=config["input"]["volume_average_days"],
                    coin_filter=config["input"]["coin_number"])