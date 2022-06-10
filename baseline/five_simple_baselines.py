# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 9:49
# @Author  : Shiqi Wang
# @FileName: five_simple_baselines.py
import argparse
import datetime
import time
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))

# sys.path.append(".")
import logzero
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
        user_active_rank_res = self.dataset.user_activity_ranks()
        user_active_list = [x for x, _ in user_active_rank_res]
        # self.active_users = user_active_list[:self.action_num]
        # self.inactive_users = user_active_list[-self.action_num:]
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

    # path = os.path.join("../config", args.dataset)
    path = '../config/config_movielens1M'
    config = configparser.ConfigParser()
    _x = open(path)

    # _x = open('../config/config_movielens100k')
    # _x = open('../config/config_movielens1M')
    # _x = open('../config/config_Ciao')
    config.read_file(_x)
    dataset = DataProcess(config)
    print("Output dir is {}".format(config['ENV']['OUT_PUT']))
    # logs
    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(config['ENV']['OUT_PUT'], "baseline_logs", "[BASELINE]_{}.log".format(nowtime))
    logzero.logfile(logger_path)

    baselines = Baselines(config,dataset)
    # start run
    epoch = 100
    active_users_avg_rew, active_users_recnum = [], []
    inactive_users_avg_rew, inactive_users_recnum = [], []
    highrating_users_avg_rew, highrating_users_recnum = [], []
    lowrating_users_avg_rew, lowrating_users_recnum = [], []
    random_users_avg_rew, random_users_recnum = [], []


    for i in range(epoch):
        print("# {} Epoch...".format(i))
        #一次获取五个baseline的结果
        baselines.reset()
        active_users,inactive_users,highrating_users,lowrating_users,user_random_list = baselines.classify()

        avg_rew0,recnum0 = baselines.return_res(active_users)
        active_users_avg_rew.append(avg_rew0)
        active_users_recnum.append(recnum0)

        avg_rew1,recnum1 = baselines.return_res(inactive_users)
        inactive_users_avg_rew.append(avg_rew1)
        inactive_users_recnum.append(recnum1)

        avg_rew2,recnum2 = baselines.return_res(highrating_users)
        highrating_users_avg_rew.append(avg_rew2)
        highrating_users_recnum.append(recnum2)

        avg_rew3, recnum3 = baselines.return_res(lowrating_users)
        lowrating_users_avg_rew.append(avg_rew3)
        lowrating_users_recnum.append(recnum3)

        avg_rew4, recnum4 = baselines.return_res(user_random_list)
        random_users_avg_rew.append(avg_rew3)
        random_users_recnum.append(recnum3)

    logzero.logger.info("average reward of active_users:{}".format(active_users_avg_rew))
    logzero.logger.info("recnum reward of active_users:{}".format(active_users_recnum))

    logzero.logger.info("average reward of inactive_users:{}".format(inactive_users_avg_rew))
    logzero.logger.info("recnum reward of inactive_users:{}".format(inactive_users_recnum))

    logzero.logger.info("average reward of highrating_users:{}".format(highrating_users_avg_rew))
    logzero.logger.info("recnum reward of hihgrating_users:{}".format(highrating_users_recnum))

    logzero.logger.info("average reward of lowrating_users:{}".format(lowrating_users_avg_rew))
    logzero.logger.info("recnum reward of lowrating_users:{}".format(lowrating_users_recnum))

    logzero.logger.info("average reward of random:{}".format(random_users_avg_rew))
    logzero.logger.info("recnum reward of ransom:{}".format(random_users_recnum))