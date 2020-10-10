#!/usr/bin/env python

import logging
from src.trainer import Trainer
from config import config
import os

def train_one(save_path, config, log_file_dir, index, logfile_level, console_level):
    """
    train an agent
    :param save_path: the path to save the tensorflow model (.ckpt), could be None
    :param config: the json configuration file
    :param log_file_dir: the directory to save the tensorboard logging file, could be None
    :param index: identifier of this train, which is also the sub directory in the train_package,
    if it is 0. nothing would be saved into the summary file.
    :param logfile_level: logging level of the file
    :param console_level: logging level of the console
    :param device: 0 or 1 to show which gpu to use, if 0, means use cpu instead of gpu
    :return : the Result namedtuple
    """
    if log_file_dir:
        logging.basicConfig(filename=log_file_dir.replace("tensorboard","programlog"),
                            level=logfile_level)
        console = logging.StreamHandler()
        console.setLevel(console_level)
        logging.getLogger().addHandler(console)
    print("training at %s started" % index)
    return Trainer(config, save_path=save_path).train(log_file_dir=log_file_dir)

train_dir = "train_package"
if not os.path.exists("./" + train_dir): 
    os.makedirs("./" + train_dir)

train_one(
    save_path="./" + train_dir + "/model", 
    config=config, 
    log_file_dir="./" + train_dir + "/tensorboard", 
    index="0", 
    logfile_level=logging.DEBUG, 
    console_level=logging.INFO
    )


