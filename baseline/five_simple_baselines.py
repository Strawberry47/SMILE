# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 9:49
# @Author  : Shiqi Wang
# @FileName: five_simple_baselines.py
import argparse
import datetime
import logging
import time
import sys
import os

import logzero

sys.path.append(os.path.dirname(sys.path[0]))

# sys.path.append(".")
from logzero import setup_logger
from smile_main.RewardCalculator import RewardCalculator
import configparser
import random
from ProcessingData.dataProcess import DataProcess


class Baselines():
    def __init__(self,config,dataset):
        self.config = config
        self.dataset = dataset
        self.action_num = int(self.config['META']['EPISODE_LENGTH'])

    def reset(self):
        self.promoted_items = random.sample(list(range(self.dataset.item_num)),
                                            int(self.dataset.item_num * 0.01))
        self.reward_calculator = RewardCalculator(self.config, self.dataset,
                                                  self.promoted_items)
        self.last_reward = self.reward_calculator.last_reward
        self.last_interactions = self.dataset.rating

    def classify(self):
        # self.action_num = test_num
        user_active_rank_res = self.dataset.user_activity_ranks()
        user_active_list = [x for x, _ in user_active_rank_res]
        self.active_users = random.sample(user_active_list[:int(0.3*self.dataset.user_num)], self.action_num)
        self.inactive_users = random.sample(user_active_list[-int(0.3 * self.dataset.user_num):], self.action_num)

        user_rating_rank_res = self.dataset.user_rating_ranks()
        user_rating_list = [x for x, _ in user_rating_rank_res]
        self.highrating_users = random.sample(user_rating_list[:int(0.3 * self.dataset.user_num):], self.action_num)
        self.lowrating_users = random.sample(user_rating_list[-int(0.3 * self.dataset.user_num):], self.action_num)
        # self.highrating_users = user_rating_list[:self.action_num]
        # self.lowrating_users = user_rating_list[-self.action_num:]
        self.user_random_list = random.sample(list(range(self.dataset.user_num)),self.action_num)
        return self.active_users,self.inactive_users,self.highrating_users,self.lowrating_users,self.user_random_list


    def return_res(self,actions):
        pos_user_list = self.reward_calculator.userfilter(actions)
        new_interactions = self.reward_calculator.insertData(self.last_interactions,pos_user_list)
        self.topnlist = self.reward_calculator.train_test(new_interactions)

        now_reward = self.reward_calculator.getItemCnt(self.topnlist)
        reward = now_reward - self.last_reward
        avg_rew = reward/len(actions)
        return round(avg_rew, 3),round(now_reward, 3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="config_movielens100k")
    args = parser.parse_known_args()[0]

    path = os.path.join("../config", args.dataset)
    # path = '../config/config_Ciao'
    # path = '../config/config_movielens1M'
    # path = '../config/config_movielens100k'
    config = configparser.ConfigParser()
    _x = open(path)

    config.read_file(_x)
    dataset = DataProcess(config)
    print("Five simple baselines, output dir is {}".format(config['ENV']['OUT_PUT']))
    # logs
    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")

    logger_path1 = os.path.join(config['ENV']['OUT_PUT'], "baseline_logs", "[active_BASELINE]_{}.log".format(nowtime))
    logger1 = setup_logger(name="mylogger1", logfile=logger_path1)


    logger_path2 = os.path.join(config['ENV']['OUT_PUT'], "baseline_logs", "[inactive_BASELINE]_{}.log".format(nowtime))
    logger2 = setup_logger(name="mylogger2", logfile=logger_path2)

    logger_path3 = os.path.join(config['ENV']['OUT_PUT'], "baseline_logs", "[highrating_BASELINE]_{}.log".format(nowtime))
    logger3 = setup_logger(name="mylogger3", logfile=logger_path3)

    logger_path4 = os.path.join(config['ENV']['OUT_PUT'], "baseline_logs", "[lowrating_BASELINE]_{}.log".format(nowtime))
    logger4 = setup_logger(name="mylogger4", logfile=logger_path4)

    logger_path5 = os.path.join(config['ENV']['OUT_PUT'], "baseline_logs", "[random_BASELINE]_{}.log".format(nowtime))
    logger5 = setup_logger(name="mylogger5", logfile=logger_path5)

    baselines = Baselines(config,dataset)
    # start run
    epochs = 100
    active_users_avg_rew, active_users_recnum = [], []
    inactive_users_avg_rew, inactive_users_recnum = [], []
    highrating_users_avg_rew, highrating_users_recnum = [], []
    lowrating_users_avg_rew, lowrating_users_recnum = [], []
    random_users_avg_rew, random_users_recnum = [], []


    for i in range(epochs):
        # num = 500
        epoch = i
        print("# {} Epoch...".format(i))
        #一次获取五个baseline的结果
        baselines.reset()
        active_users,inactive_users,highrating_users,lowrating_users,user_random_list = baselines.classify()

        avg_rew0,recnum0 = baselines.return_res(active_users)
        active_users_avg_rew.append(avg_rew0)
        active_users_recnum.append(recnum0)
        result = {'ave_rew': avg_rew0, 'recnum': recnum0}
        logger1.info("Epoch: [{}], Info: [{}]".format(epoch, result))

        avg_rew1,recnum1 = baselines.return_res(inactive_users)
        inactive_users_avg_rew.append(avg_rew1)
        inactive_users_recnum.append(recnum1)
        result = {'ave_rew': avg_rew1, 'recnum': recnum1}
        logger2.info("Epoch: [{}], Info: [{}]".format(epoch, result))

        avg_rew2,recnum2 = baselines.return_res(highrating_users)
        highrating_users_avg_rew.append(avg_rew2)
        highrating_users_recnum.append(recnum2)
        result = {'ave_rew': avg_rew2, 'recnum': recnum2}
        logger3.info("Epoch: [{}], Info: [{}]".format(epoch, result))

        avg_rew3, recnum3 = baselines.return_res(lowrating_users)
        lowrating_users_avg_rew.append(avg_rew3)
        lowrating_users_recnum.append(recnum3)
        result = {'ave_rew': avg_rew3, 'recnum': recnum3}
        logger4.info("Epoch: [{}], Info: [{}]".format(epoch, result))

        avg_rew4, recnum4 = baselines.return_res(user_random_list)
        random_users_avg_rew.append(avg_rew3)
        random_users_recnum.append(recnum3)
        result = {'ave_rew': avg_rew3, 'recnum': recnum3}
        logger5.info("Epoch: [{}], Info: [{}]".format(epoch, result))

    # logzero.logger.info("average reward of active_users:{}".format(active_users_avg_rew))
    # logzero.logger.info("recnum reward of active_users:{}".format(active_users_recnum))
    #
    # logzero.logger.info("average reward of inactive_users:{}".format(inactive_users_avg_rew))
    # logzero.logger.info("recnum reward of inactive_users:{}".format(inactive_users_recnum))
    #
    # logzero.logger.info("average reward of highrating_users:{}".format(highrating_users_avg_rew))
    # logzero.logger.info("recnum reward of hihgrating_users:{}".format(highrating_users_recnum))
    #
    # logzero.logger.info("average reward of lowrating_users:{}".format(lowrating_users_avg_rew))
    # logzero.logger.info("recnum reward of lowrating_users:{}".format(lowrating_users_recnum))
    #
    # logzero.logger.info("average reward of random:{}".format(random_users_avg_rew))
    # logzero.logger.info("recnum reward of ransom:{}".format(random_users_recnum))

    # result = {'ave_rew':avg_rew,'recnum':recnum}
    # logzero.logger.info("Epoch: [{}], Info: [{}]".format(epoch, result))