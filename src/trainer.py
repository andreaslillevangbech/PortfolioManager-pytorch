import datetime
import os
import logging
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.data.datamatrices import DataMatrices
from src.agent import Agent


class Trainer:
    def __init__(self, config, agent = None, save_path = None, restore_dir = None, device = "cpu"):
        self.config = config
        self.train_config = config["training"]
        self.input_config = config["input"]
        self.best_metric = 0
        self.save_path = save_path

        self._matrix = DataMatrices.create_from_config(config)
        self.time_index = self._matrix._DataMatrices__global_data.time_index.values
        self.coins = self._matrix._DataMatrices__global_data.coins.values
        self.test_set = self._matrix.get_test_set()
        self.training_set = self._matrix.get_training_set()

        torch.random.manual_seed(config["random_seed"])
        self.device = device
        self._agent = Agent(
            config,
            time_index=self.time_index, 
            coins=self.coins, 
            restore_dir=restore_dir,
            device=device)

    def __init_tensorboard(self, log_file_dir = 'logs/'):
        # current_time = datetime.datetime.now().strftime("%Y%m%d-H%M%S")
        train_log_dir = os.path.join(log_file_dir, 'train')
        test_log_dir = os.path.join(log_file_dir, 'test')
        self.train_writer = SummaryWriter(train_log_dir)
        self.test_writer = SummaryWriter(test_log_dir)
        network_writer = SummaryWriter(os.path.join(log_file_dir, 'graph'))
        X, w, _, _ = self.next_batch()
        network_writer.add_graph(self._agent.model, [X, w])
        network_writer.close()

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

    def _evaluate(self, set_name):
        if set_name=="test":
            batch = self.test_set
        elif set_name == "training":
            batch = self.training_set
        else:
            raise ValueError()
        w = torch.tensor(batch['last_w'])
        w = w[:, None, : , None] # Concat along dim=1, the features dim)
        X = torch.tensor(batch['X'])
        y = torch.tensor(batch['y']) 
        pv_vector, loss, output = self._agent.evaluate(X, w, y)
        return pv_vector, loss, output

    def log_between_steps(self, step):
        fast_train = self.train_config["fast_train"]

        # Summary on test set. Evaluating the agent updates the agents metrics
        pv_vector, v_loss, v_output = self._evaluate("test")
        # Get some stats
        v_pv = self._agent.portfolio_value
        v_log_mean = self._agent.log_mean
        log_mean_free = self._agent.log_mean_free

        self.test_writer.add_scalar('portfolio value', self._agent.portfolio_value, global_step=step)
        self.test_writer.add_scalar('mean', self._agent.mean, global_step=step)
        self.test_writer.add_scalar('log_mean', self._agent.log_mean, global_step=step)
        self.test_writer.add_scalar('std', self._agent.standard_deviation, global_step=step)
        self.test_writer.add_scalar('loss', v_loss,global_step=step)
        self.test_writer.add_scalar("log_mean_free", self._agent.log_mean_free,global_step=step)
        for name, param in self._agent.model.named_parameters():
            self.test_writer.add_histogram(name, param, global_step=step)

        # Save model
        if v_pv > self.best_metric:
            self.best_metric = v_pv
            logging.info("get better model at %s steps,"
                         " whose test portfolio value is %s" % (step, self._agent.portfolio_value))
            if self.save_path:
                torch.save(self._agent.model.state_dict(), self.save_path)
                # self._agent.model.save_weights(self.save_path)

        if not fast_train:
            pv_vector, loss, output = self._evaluate("training")
            self.train_writer.add_scalar('portfolio value', self._agent.portfolio_value,global_step=step)
            self.train_writer.add_scalar('mean', self._agent.mean,global_step=step)
            self.train_writer.add_scalar('log_mean', self._agent.log_mean,global_step=step)
            self.train_writer.add_scalar('std', self._agent.standard_deviation,global_step=step)
            self.train_writer.add_scalar('loss', loss,global_step=step)
            self.train_writer.add_scalar("log_mean_free", self._agent.log_mean_free,global_step=step)
            for name, param in self._agent.model.named_parameters():
                self.train_writer.add_histogram(name, param,global_step=step)
            
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

        # Dunno what this is for.
        self.check_abnormal(self._agent.portfolio_value, output)

    def check_abnormal(self, portfolio_value, weigths):
        if portfolio_value == 1.0:
            logging.info("average portfolio weights {}".format(weigths.mean(axis=0)))

    def next_batch(self):
        batch = self._matrix.next_batch()
        w = torch.tensor(batch['last_w'])
        w = w[:, None, : , None] # Concat along dim=1, the features dim)
        X = torch.tensor(batch['X'])
        y = torch.tensor(batch['y'])
        return X, w, y, batch['setw']

    def train(self, log_file_dir = "./tensorboard", index = "0"):

        self.__print_upperbound()
        self.__init_tensorboard(log_file_dir)
        
        starttime = time.time()
        total_data_time = 0
        total_training_time = 0
        for i in range(int(self.train_config['steps'])):
            step_start = time.time()
            X, w, y, setw = self.next_batch()
            finish_data = time.time()
            total_data_time += (finish_data - step_start)
            self._agent.train_step(X, w, y, setw=setw)
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

        pv_vector, loss, output = self._evaluate("test")
        pv = self._agent.portfolio_value
        log_mean = self._agent.log_mean
        logging.warning('the portfolio value train No.%s is %s log_mean is %s,'
                        ' the training time is %d seconds' % (index, pv, log_mean, time.time() - starttime))
