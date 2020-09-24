import datetime
import tensorflow as tf
import logging
import time
import numpy as np

from src.data.datamatrices import DataMatrices
from src.agent import Agent


# NOTES
# Custom metrics and loss for use in tensorboard and possibly the fit and evaluate tf functions

class Trainer:
    def __init__(self, config, agent = None, save_path = None, restore_dir = None, device = "cpu"):
        self.config = config
        self.train_config = config["training"]
        self.input_config = config["input"]
        self.best_metric = 0
        self.save_path = save_path

        self._matrix = DataMatrices.create_from_config(config)
        self.test_set = self._matrix.get_test_set()
        self.training_set = self._matrix.get_training_set()

        self._agent = Agent(config, restore_dir=restore_dir)


    def init_tensorboard(self, log_file_dir = 'logs/'):
        current_time = datetime.datetime.now().strftime("%Y%m%d-H%M%S")
        train_log_dir = log_file_dir + '/' + current_time + '/train'
        test_log_dir = log_file_dir + '/' + current_time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    @staticmethod
    def calculate_upperbound(y):
        array = np.maximum.reduce(y[:, 0, :], 1)
        total = 1.0
        for i in array:
            total = total * i
        return total

    def __print_upperbound(self):
        upperbound_test = self.calculate_upperbound(self.test_set["y"])
        logging.info("upper bound in test is %s" % upperbound_test)

    def log_between_steps(self, step):
        fast_train = self.train_config["fast_train"]

        # Summary on test set. Evaluating the agent updates the agents metrics
        pv_vector, v_loss, v_output = self._agent.evaluate(self.test_set)
        # Get some stats
        v_pv = self._agent.portfolio_value
        v_log_mean = self._agent.log_mean
        log_mean_free = self._agent.log_mean_free

        with self.test_summary_writer.as_default() as writer:
            tf.summary.scalar('portfolio value', self._agent.portfolio_value, step=step)
            tf.summary.scalar('mean', self._agent.mean, step=step)
            tf.summary.scalar('log_mean', self._agent.log_mean, step=step)
            tf.summary.scalar('std', self._agent.standard_deviation, step=step)
            tf.summary.scalar('loss', v_loss, step=step)
            tf.summary.scalar("log_mean_free", self._agent.log_mean_free, step=step)
            writer.flush()

        # NOTE: add summary for training set too.
        if not fast_train:
            pv_vector, loss, output = self._agent.evaluate(self.training_set)
            with self.train_summary_writer.as_default() as writer:
                tf.summary.scalar('portfolio value', self._agent.portfolio_value, step=step)
                tf.summary.scalar('mean', self._agent.mean, step=step)
                tf.summary.scalar('log_mean', self._agent.log_mean, step=step)
                tf.summary.scalar('std', self._agent.standard_deviation, step=step)
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar("log_mean_free", self._agent.log_mean_free, step=step)
                writer.flush()

        # print 'ouput is %s' % out
        logging.info('='*30)
        logging.info('step %d' % step)
        logging.info('-'*30)
        if not fast_train:
            logging.info('training loss is %s\n' % loss)
        logging.info('the portfolio value on test set is %s\nlog_mean is %s\n'
                     'loss_value is %3f\nlog mean without commission fee is %3f\n' % \
                     (v_pv, v_log_mean, v_loss, log_mean_free))
        logging.info('='*30+"\n")


        # NOTE: Save model
        if v_pv > self.best_metric:
            self.best_metric = v_pv
            logging.info("get better model at %s steps,"
                         " whose test portfolio value is %s" % (step, self._agent.portfolio_value))
            if self.save_path:
                self._agent.model.save_weights(self.save_path)
        
        # Dunno what this is for.
        self.check_abnormal(self._agent.portfolio_value, output)

    def check_abnormal(self, portfolio_value, weigths):
        if portfolio_value == 1.0:
            logging.info("average portfolio weights {}".format(weigths.mean(axis=0)))

    def train(self, log_file_dir = "./tensorboard", index = "0"):
        #loss_metric = -tf.keras.metrics.Mean()

        self.__print_upperbound()
        self.init_tensorboard(log_file_dir)
        
        starttime = time.time()
        total_data_time = 0
        total_training_time = 0
        for i in range(self.train_config['steps']):
            step_start = time.time()
            batch = self._matrix.next_batch()
            finish_data = time.time()
            total_data_time += (finish_data - step_start)
            # Do a train step
            self._agent.train_step(batch)
            total_training_time += time.time() - finish_data 
            if i % 1000 == 0 and log_file_dir:
                logging.info("average time for data accessing is %s"%(total_data_time/1000))
                logging.info("average time for training is %s"%(total_training_time/1000))
                total_training_time = 0
                total_data_time = 0
                self.log_between_steps(i)
            
        if self.save_path:
            best_agent = Agent(self.config, restore_dir=self.save_path)
            self._agent = best_agent

        pv_vector, loss, output = self._agent.evaluate(self.test_set)
        pv = self._agent.portfolio_value
        log_mean = self._agent.log_mean
        logging.warning('the portfolio value train No.%s is %s log_mean is %s,'
                        ' the training time is %d seconds' % (index, pv, log_mean, time.time() - starttime))