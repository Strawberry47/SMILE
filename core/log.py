# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 0:56
# @Author  : Shiqi Wang
# @FileName: log.py
# -*- coding: utf-8 -*-
# @Time    : 2021/8/2 7:56 下午
# @Author  : Chongming GAO
# @FileName: utils.py


import logzero
from logzero import logger
import os
from tensorflow.python.keras.callbacks import Callback



def create_dir(create_dirs):
    """
    创建所需要的目录
    """
    for dir in create_dirs:
        if not os.path.exists(dir):
            logger.info('Create dir: %s' % dir)
            try:
                os.mkdir(dir)
            except FileExistsError:
                print("The dir [{}] already existed".format(dir))


# my nas docker path
REMOTE_ROOT = "/root/Counterfactual_IRS"
# 回调函数
class LoggerCallback_Update(Callback):
    def __init__(self, logger_path):
        super().__init__()
        self.LOCAL_PATH = logger_path
        self.REMOTE_ROOT = REMOTE_ROOT
        self.REMOTE_PATH = os.path.join(REMOTE_ROOT, os.path.dirname(logger_path))

    def on_epoch_end(self, epoch, logs=None):
        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, logs))

        # 2. upload logger
        # self.upload_logger()

    # def upload_logger(self):
    #     try:
    #         my_upload(self.LOCAL_PATH, self.REMOTE_PATH, self.REMOTE_ROOT)
    #     except Exception:
    #         print("Failed: Uploading file [{}] to remote path [{}]".format(self.LOCAL_PATH, self.REMOTE_PATH))


class LoggerCallback_RL(LoggerCallback_Update):
    def __init__(self, logger_path):
        super().__init__(logger_path)

    def on_epoch_end(self, epoch, logs=None):
        num_test = logs["n/ep"]
        # len_tra = logs["n/st"] / num_test
        best_average_rew = logs["best_average_rew"]
        best_recnum = logs["best_recnum"]

        result = dict()
        result['num_test'] = num_test
        result['ave_rew'] = best_average_rew
        result['recnum'] = best_recnum


        # 1. write logger
        logger.info("Epoch: [{}], Info: [{}]".format(epoch, result))

        # 2. upload logger
        # self.upload_logger()
