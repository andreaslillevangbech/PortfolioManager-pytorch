import src.data.globaldatamatrix as gdm
from src.data.replayBuffer import ReplayBuffer
import numpy as np
import pandas as pd
import logging
from src.tools.configprocess import parse_time
from src.tools.data import get_volume_forward, get_type_list

MIN_NUM_PERIOD = 3


class DataMatrices:
    def __init__(self, start, end, period, batch_size=50, volume_average_days=30, buffer_bias=0,
                 market="poloniex", coin_filter=1, window_size=50, feature_number=3, test_portion=0.15,
                 portion_reversed=False, online=False ):
        """
        :param start: Unix time
        :param end: Unix time
        :param access_period: the data access period of the input matrix.
        :param trade_period: the trading period of the agent.
        :param global_period: the data access period of the global price matrix.
                              if it is not equal to the access period, there will be inserted observations
        :param coin_filter: number of coins that would be selected
        :param window_size: periods of input data
        :param train_portion: portion of training set
        :param is_permed: if False, the sample inside a mini-batch is in order
        :param validation_portion: portion of cross-validation set
        :param test_portion: portion of test set
        :param portion_reversed: if False, the order to sets are [train, validation, test]
        else the order is [test, validation, train]
        """
        start = int(start)
        self.__end = int(end)

        # assert window_size >= MIN_NUM_PERIOD
        self.__coin_no = coin_filter
        type_list = get_type_list(feature_number)
        self.__features = type_list
        self.feature_number = feature_number
        volume_forward = get_volume_forward(self.__end-start, test_portion, portion_reversed)
        self.__history_manager = gdm.HistoryManager(coin_number=coin_filter, end=self.__end,
                                                    volume_average_days=volume_average_days,
                                                    volume_forward=volume_forward, online=online)
        if market == "poloniex":
            self.__global_data = self.__history_manager.get_global_panel(start,
                                                                         self.__end,
                                                                         period=period,
                                                                         features=type_list)
        else:
            raise ValueError("market {} is not valid".format(market))
        self.__period_length = period

        # portfolio vector memory, [time, assets]
        PVM = pd.DataFrame(index=self.__global_data.time_index.values, 
                                    columns=self.__global_data.coins.values, dtype='float32')
        self.__PVM = PVM.fillna(1.0 / self.__coin_no)

        self._window_size = window_size
        self._no_periods = len(self.__global_data.time_index.values)
        self.divide_data(test_portion) # This gives the indekses of the training and test data

        self._portion_reversed = portion_reversed

        self.__batch_size = batch_size
        self.__delta = 0  # the count of global increased


        self.__buffer_bias = buffer_bias
        # This needs to be written such that it gets arguments from config, like sample bias (geo dist)
        end_index = self._train_ind[-1]
        self.__replay_buffer = ReplayBuffer(start_index=self._train_ind[0],
                                               end_index=end_index,
                                               sample_bias=self.__buffer_bias,
                                               batch_size=self.__batch_size,
                                               coin_no=self.__coin_no)

        logging.info("the number of training examples is %s"
                     ", of test examples is %s" % (self._num_train_samples, self._num_test_samples))
        logging.debug("the training set is from %s to %s" % (min(self._train_ind), max(self._train_ind)))
        logging.debug("the test set is from %s to %s" % (min(self._test_ind), max(self._test_ind)))


    @property
    def global_weights(self):
        return self.__PVM

    @staticmethod
    def create_from_config(config):
        """main method to create the DataMatrices in this project
        @:param config: config dictionary
        @:return: a DataMatrices object
        """
        config = config.copy()
        input_config = config["input"]
        train_config = config["training"]
        start = parse_time(input_config["start_date"])
        end = parse_time(input_config["end_date"])
        return DataMatrices(start=start,
                            end=end,
                            market=input_config["market"],
                            feature_number=input_config["feature_no"],
                            window_size=input_config["window_size"],
                            online=input_config["online"],
                            period=input_config["global_period"],
                            coin_filter=input_config["coin_no"],
                            buffer_bias = train_config['buffer_bias'],
                            batch_size = train_config["batch_size"],
                            volume_average_days=input_config["volume_average_days"],
                            test_portion=input_config["test_portion"]
                            )

    @property
    def global_matrix(self):
        return self.__global_data

    @property
    def coin_list(self):
        return self.__history_manager.coins

    @property
    def num_train_samples(self):
        return self._num_train_samples

    @property
    def num_test_samples(self):
        return self._num_test_samples

    def append_experience(self, online_w=None):
        """
        :param online_w: (number of assets + 1, ) numpy array
        Let it be None if in the backtest case.
        """
        self.__delta += 1
        self._train_ind.append(self._train_ind[-1]+1)
        appended_index = self._train_ind[-1]
        self.__replay_buffer.append_experience(appended_index)

    def keras_batch(self, data=None):
        if data=="test":
            indexs = self._test_ind[(self._window_size):] 
        else:
            indexs = [exp.state_index for exp in self.__replay_buffer.next_experience_batch()] 
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M) 
        X = M[:, :, :, :-1]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]
        return {"X": X, "y": y, "idx": indexs}

    def next_batch(self):
        """
        @:return: the next batch of training sample. The sample is a dictionary
        with key "X"(input data); "y"(future relative price); "last_w" a numpy array
        with shape [batch_size, assets]; "w" a list of numpy arrays list length is
        batch_size
        """
        batch = self.pack_samples([exp.state_index for exp in self.__replay_buffer.next_experience_batch()])
        return batch

    def pack_samples(self, indexs):
        indexs = np.array(indexs)
        last_w = self.__PVM.values[indexs-1, :]

        def setw(w):                      # Notice that this function is defined in terms of the specifik indexs
            self.__PVM.iloc[indexs, :] = w    
        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M) 
        # M is features, coins, time
        X = M[:, :, :, :-1] 
        y = M[:, :, :, -1] / M[:, 0, None, :, -2]     # y_{t+1} obtained by dividing all features by prev close price
        return {"X": X, "y": y, "last_w": last_w, "setw": setw}

    # volume in y is the volume in next access period
    def get_submatrix(self, ind):
        return self.__global_data.values[:, :, ind-(self._window_size):ind+1]

    def get_test_set(self):
        return self.pack_samples(self._test_ind[(self._window_size+1):]) # Make sure you dont use prices in the test sample that were used in training

    def get_training_set(self):
        return self.pack_samples(self._train_ind)

    def divide_data(self, test_portion, portion_reversed = False):
        train_portion = 1 - test_portion
        s = float(train_portion + test_portion)
        if portion_reversed:
            portions = np.array([test_portion]) / s
            portion_split = (portions * self._no_periods).astype(int)
            indices = np.arange(self._no_periods)
            self._test_ind, self._train_ind = np.split(indices, portion_split)
        else:
            portions = np.array([train_portion]) / s
            portion_split = (portions * self._no_periods).astype(int)
            indices = np.arange(self._no_periods)
            self._train_ind, self._test_ind = np.split(indices, portion_split)

        self._train_ind = self._train_ind[(self._window_size):]
        # NOTE(zhengyao): change the logic here in order to fit both
        # reversed and normal version
        self._train_ind = list(self._train_ind)
        self._num_train_samples = len(self._train_ind)
        self._num_test_samples = len(self._test_ind)
