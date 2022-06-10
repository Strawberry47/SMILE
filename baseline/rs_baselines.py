# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 15:28
# @Author  : Shiqi Wang
# @FileName: rs_baselines.py
# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 9:49
# @Author  : Shiqi Wang
# @FileName: five_simple_baselines.py
import argparse
import datetime
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from collections import Counter
from core import run_time_tools
import time
from sklearn.neighbors import NearestNeighbors
import logzero
from smile_main.RewardCalculator import RewardCalculator
import configparser
import random
from ProcessingData.dataProcess import DataProcess
import numpy as np

class Baselines():
    def __init__(self,config,dataset):
        self.config = config
        self.dataset = dataset
        self.rating_df = dataset.rating
        self.action_num = int(self.config['META']['EPISODE_LENGTH'])

        user_embedding_path = config['ENV']['OUT_PUT'] + 'user_embedding_dim%d' % int(config['META']['ACTION_DIM'])
        item_embedding_path = config['ENV']['OUT_PUT'] + 'item_embedding_dim%d' % int(config['META']['ITEM_DIM'])

        if not os.path.exists(user_embedding_path):
            run_time_tools.mf_with_bias(self.config, dataset)  # 利用PMF
        self.user_embedding = np.loadtxt(user_embedding_path, dtype=float, delimiter='\t')
        self.item_embedding = np.loadtxt(item_embedding_path, dtype=float, delimiter='\t')


    def reset(self):
        self.promoted_items = random.sample(list(range(self.dataset.item_num)),
                                            int(self.dataset.item_num * 0.01))
        self.reward_calculator = RewardCalculator(self.config, self.dataset,
                                                  self.promoted_items)
        self.last_reward = self.reward_calculator.last_reward
        self.last_interactions = self.dataset.rating

    def nearestItems(self):
        k_nearest = set()
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(self.item_embedding)
        # find the nearest 20 items to construct res
        i = 0
        while len(k_nearest)<len(self.promoted_items):
            cur_item_index = self.promoted_items[i]
            indices = nbrs.kneighbors([self.item_embedding[cur_item_index]], return_distance=False)
            k_nearest.update(indices[0][1:].tolist())
            i = i+1
        return list(k_nearest)

    def item_cf(self):
        similar_item = self.nearestItems()
        index_list = []
        # 找到
        for i in range(len(similar_item)):
            indice = self.rating_df[self.rating_df['itemid'] == similar_item[i]]['userid'].values
            index_list.append(indice.tolist())
        count = Counter(sum(index_list,[]))
        user_list = sorted(count, key=count.get, reverse=True)
        res = random.sample(user_list, self.action_num)
        return res

#### 先找到promoted item的用户，再找相似用户
    def buyers(self):
        # find the buyers of items
        target_item = self.promoted_items
        index_list = []
        # 找到
        for i in range(len(target_item)):
            indice = self.rating_df[self.rating_df['itemid'] == target_item[i]]['userid'].values
            index_list.append(indice.tolist())
        count = Counter(sum(index_list, []))
        user_list = sorted(count, key=count.get, reverse=True)
        return user_list[:self.action_num]

    def user_cf(self):
        # 10 buyers
        user_list = self.buyers()
        k_nearest = set()
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(self.user_embedding)
        # find the nearest 20 items to construct res
        i = 0
        while len(k_nearest)<self.action_num*2:
            cur_user_index = user_list[i]
            indices = nbrs.kneighbors([self.user_embedding[cur_user_index]], return_distance=False)
            k_nearest.update(indices[0][1:].tolist())
            i = i+1

        res = random.sample(list(k_nearest), self.action_num)
        return res



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
    logger_path = os.path.join(config['ENV']['OUT_PUT'], "baseline_logs", "[CF_BASELINES]_{}.log".format(nowtime))
    logzero.logfile(logger_path)

    baselines = Baselines(config,dataset)
    # start run
    epoch = 100
    usercf_avg_rew, usercf_recnum = [], []
    itemcf_avg_rew, itemcf_recnum = [], []

    for i in range(epoch):
        print("# {} Epoch...".format(i))
        #一次获取五个baseline的结果
        baselines.reset()
        itemcf_users = baselines.item_cf()
        usercf_users = baselines.user_cf()

        avg_rew0,recnum0 = baselines.return_res(itemcf_users)
        usercf_avg_rew.append(avg_rew0)
        usercf_recnum.append(recnum0)

        avg_rew1,recnum1 = baselines.return_res(usercf_users)
        itemcf_avg_rew.append(avg_rew1)
        itemcf_recnum.append(recnum1)


    logzero.logger.info("average reward of usercf:{}".format(usercf_avg_rew))
    logzero.logger.info("recnum reward of usercf:{}".format(usercf_recnum))

    logzero.logger.info("average reward of itemcf:{}".format(itemcf_avg_rew))
    logzero.logger.info("recnum reward of itemcf:{}".format(itemcf_recnum))
