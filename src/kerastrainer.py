import datetime
import tensorflow as tf
import logging
import time
import numpy as np
import pandas as pd

from src.data.datamatrices import DataMatrices
from src.network import CNN_Keras

class KerasTrainer:
    def __init__(self, config, save_path = None, restore_dir = None, device = "cpu"):
        self.best_metric = 0
        self.save_path = save_path

        self.config = config
        self.layers = config["layers"]
        self.train_config = config["training"]
        self.input_config = config["input"]

        self.batch_size = self.train_config['batch_size']
        self.learning_rate = self.train_config['learning_rate']
        
        self.coin_no =self.input_config['coin_no']
        self.window_size = self.input_config['window_size']
        self.global_period = self.input_config["global_period"]
        self.feature_no = self.input_config['feature_no']
        self.commission_ratio = config['trading']["trading_consumption"]

        self._matrix = DataMatrices.create_from_config(config)
        self.time_index = self._matrix._DataMatrices__global_data.time_index.values
        self.coins = self._matrix._DataMatrices__global_data.coins.values

        tf.random.set_seed(self.config["random_seed"])
        self.device = device


        PVM = pd.DataFrame(index=self.time_index, 
                                    columns=self.coins, dtype='float32')
        self.__PVM = PVM.fillna(1.0 / len(self.coins))

        self.model = CNN_Keras(
            self.feature_no,
            self.coin_no, 
            self.window_size,
            self.layers,
            self.__PVM.values
        )

        # initial_learning_rate = self.learning_rate
        initial_learning_rate = 0.1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, 
            decay_steps=100000, 
            decay_rate=0.96, 
            staircase=True
        )

        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(self.learning_rate),
            loss = self.loss,

        )
        # self.model.run_eagerly = True
        # logging.INFO("run_eagerly set to: %s" % self.model.run_eagerly )

    
    
    def keras_val(self):
        batch = self._matrix.keras_batch(data="test")
        X = tf.transpose(batch['X'], [0, 2, 3, 1]) 
        y = batch['y'][:, 0, :]
        idx = np.array(batch['idx'], dtype='int32')
        return ([X, idx], y) 

    def keras_gen(self):
        self.global_step = 0
        while True:
            batch = self._matrix.keras_batch()
            X = tf.transpose(batch['X'], [0, 2, 3, 1]) 
            y = batch['y'][:, 0, :]
            idx = np.array(batch['idx'], dtype='int32')
            yield ([X, idx], y)
            self.global_step +=1

    def keras_fit(self, logdir='keras_train/tensorboard'):
        # self.__print_upperbound()
        tensorboard_callback= tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=1
        )
        self.history = self.model.fit(
            x = self.keras_gen(),
            callbacks = [tensorboard_callback],
            validation_data = self.keras_val(),
            steps_per_epoch = int(self.train_config['steps'])//100,
            epochs=100
        )

    def loss(self, y, output):
        #r_t = log(mu_t * y_t dot w_{t-1})
        input_no = tf.shape(y)[0]
        future_price = tf.concat([tf.ones([input_no, 1]), y], 1) # Add cash price (always 1)
        future_w = (future_price * output) / tf.reduce_sum(future_price * output, axis=1)[:, None]
        pv_vector = tf.reduce_sum(output * future_price, axis=1) *\
                           (tf.concat([tf.ones(1), self.pure_pc(output, input_no, future_w)], 0))
        
        return -tf.reduce_mean(tf.math.log(pv_vector))
        
    # consumption vector (on each periods)
    def pure_pc(self, output, input_no, future_w):
        c = self.commission_ratio
        w_t = future_w[:input_no-1]  # rebalanced
        w_t1 = output[1:input_no]
        mu = 1 - tf.reduce_sum(tf.math.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c
        """
        mu = 1-3*c+c**2

        def recurse(mu0):
            factor1 = 1/(1 - c*w_t1[:, 0])
            if isinstance(mu0, float):
                mu0 = mu0
            else:
                mu0 = mu0[:, None]
            factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
                tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
            return factor1*factor2

        for i in range(20):
            mu = recurse(mu)
        """
        return mu

    def __print_upperbound(self):
        upperbound_test = self.calculate_upperbound(self.keras_test["y"])
        logging.info("upper bound in test is %s" % upperbound_test)
    
    @staticmethod
    def calculate_upperbound(y):
        array = np.maximum.reduce(y[:, 0, :], 1)
        total = 1.0
        for i in array:
            total = total * i
        return total