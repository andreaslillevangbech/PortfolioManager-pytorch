#!/usr/bin/env python
import logging
from src.trainer import Trainer
from config import config
import os

def train_one(save_path, config, log_file_dir, index, logfile_level, console_level):
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


