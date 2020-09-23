import numpy as np
import pandas as pd

from src.trainer import Trainer

class Trader:
    def __init__(self, config, wait_period, total_steps, initial_BTC=1):
        self.steps = 0
        self.total_steps = total_steps
        self.wait_period = wait_period

        # the total assets is calculated with BTC
        self._total_capital = initial_BTC
        self._window_size = config["input"]["window_size"]
        self._coin_number = config["input"]["coin_no"]
        self._commission_rate = config["trading"]["trading_consumption"]
        self._asset_vector = np.zeros(self._coin_number+1)

        self._last_omega = np.zeros((self._coin_number+1,))
        self._last_omega[0] = 1.0


    def _initialize_logging_data_frame(self, initial_BTC):
        logging_dict = {'Total Asset (BTC)': initial_BTC, 'BTC': 1}
        for coin in self._coin_name_list:
            logging_dict[coin] = 0
        self._logging_data_frame = pd.DataFrame(logging_dict, index=pd.to_datetime([time.time()], unit='s'))