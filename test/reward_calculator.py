# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 13:32
# @Author  : Shiqi Wang
# @FileName: reward_calculator.py
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 22:16
# @Author  : Shiqi Wang
# @FileName: RewardCalculator.py
import configparser
import os
from collections import defaultdict, Counter
from time import time

import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ProcessingData.dataProcess import DataProcess
from core import run_time_tools, utils
from core.MF import BPR, BPRDataset
from core.utils import negative_sample


class RewardCalculator():
    def __init__(self,config,dataset,promoted_items):
        promoted_item_list = config['ENV']['TARGET_ITEM_LIST'].split(', ')
        self.config = config
        self.promoted_items = promoted_items
        self.filenameU = config['META']['filenameU']
        self.filenameI = config['META']['filenameI']
        user_embedding_path = config['ENV']['OUT_PUT'] + 'user_embedding_dim%d' % int(config['META']['ACTION_DIM'])
        item_embedding_path = config['ENV']['OUT_PUT'] + 'item_embedding_dim%d' % int(config['META']['ITEM_DIM'])

        if not os.path.exists(user_embedding_path):
            run_time_tools.mf_with_bias(self.config, dataset)  # 利用PMF
        self.user_embedding = np.loadtxt(user_embedding_path, dtype=float, delimiter='\t')
        self.item_embedding = np.loadtxt(item_embedding_path, dtype=float, delimiter='\t')

        # 原数据
        self.original_data = dataset.rating
        self.candidate_len = int(0.05*dataset.item_num)
        self.candidate = self.candidateGeneration()
        # calculate initial cnt
        topnlist = self.train_test(self.original_data)
        self.last_reward = self.getItemCnt(topnlist)


    def userfilter(self,actions):
        # return items and adopters
        positiveUserlist = defaultdict(list)
        negativeUsealist = defaultdict(list)
        itemlist = self.promoted_items

        for item in itemlist:
            for userid in actions:
                rating = round(self.user_embedding[userid].dot(self.item_embedding[item]))
                if (rating >= 0): # todo 3
                    positiveUserlist[item].append(userid)
                else:
                    negativeUsealist[item].append(userid)
        return positiveUserlist

    def insertData(self, original_data,pos_user_list):
        # return new interaction record
        new_data = original_data.values.tolist()
        insert_data = []
        # 注入连接后的新数据
        for item, user in pos_user_list.items():
            for uid in user:
                insert_data.append([uid, item, int(1)])
        new_data.extend(insert_data)

        # print("Insert lens: {}; New interactions lens: {} ".format(len(insert_data),len(new_data)))
        interaction_df = pd.DataFrame(new_data,columns = ['userid','itemid','rating'])
        ## notice!
        # self.original_data = interaction_df
        return interaction_df

    def candidateGeneration(self):
        k_nearest = set()
        k_nearest.update(self.promoted_items)
        nbrs = NearestNeighbors(n_neighbors=15, algorithm='auto').fit(self.item_embedding)
        # find the nearest 100 items to construct candidate
        i = 0
        while len(k_nearest)<self.candidate_len:
            cur_item_index = self.promoted_items[i]
            indices = nbrs.kneighbors([self.item_embedding[cur_item_index]], return_distance=False)
            k_nearest.update(indices[0].tolist())
            i = i+1
        return list(k_nearest)

    def train_test(self,df,device='cpu'):
        # BPR
        GPU = torch.cuda.is_available()
        device = torch.device('cuda' if GPU else "cpu")
        model = BPR(self.config,df,device)
        model = model.to(device)

        bpr = utils.BPRLoss(model)
        # negative sample
        users,posItems,negItems = negative_sample(df)

        users = users.to(device)
        posItems = posItems.to(device)
        negItems = negItems.to(device)

        users, posItems, negItems = utils.shuffle(users, posItems, negItems)  # 传入的是参数
        total_batch = len(users) // (2048 + 1)
        aver_loss = 0.

        with tqdm(enumerate(utils.minibatch(users, posItems, negItems, batch_size=2048)),
                  desc='BPR training',total=total_batch,leave=True) as t:
            for batch_i, (batch_users, batch_pos, batch_neg) in t:
                cri = bpr.stageOne(batch_users, batch_pos, batch_neg)  # 分批次传入(u,i,j)，更新，返回loss
                aver_loss += cri
        aver_loss = aver_loss / total_batch
        print(' avg_loss: %f' % (aver_loss))


        # train_dataset = BPRDataset(self.config,df, device)
        # train_iter = DataLoader(train_dataset, batch_size=2048)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # start training
        # for epoch in range(1):
        #     model.train()
        #     total_loss, total_len = 0.0, 0
        #     for user,item_i,item_j in train_iter:
        #         loss = model.cal_loss(user,item_i,item_j)
        #         optimizer.zero_grad()  # 清空这一批的梯度
        #         loss.backward()  # 回传
        #         optimizer.step()  # 参数更新
        #         total_loss += loss
        #         total_len += len(user)
        # #     print('----round%2d: avg_loss: %f' % (epoch, total_loss / total_len))
        # # print('BPR training done.')

        # start evulate
        model.eval()
        with torch.no_grad():
            users = list(self.original_data['userid'].unique())
            rating = model.getUsersRating(users,self.candidate)
            value, indice = torch.topk(rating, k=10)
            indice = indice.cpu().numpy()
            topk = np.array(self.candidate)[indice]
        return topk


    # 查找item在TopN列表出现的次数
    def getItemCnt(self, result):
        targetItem = self.promoted_items
        topN_list = np.array(result).flatten().tolist()
        count = {}
        for item in targetItem:
            count[item] = Counter(topN_list)[item]
        average_cnt = int(sum(count.values())) / len(targetItem)
        # print("Target item " + " ".join(str(i) for i in self.targetItem) +
        #       " appears " + str(sum(count.values())) + " times in the user TopN list.")
        # print("Avg:"+str(average_cnt))
        return round(average_cnt, 3)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    _x = open('../config/config_movielens100k')
    config.read_file(_x)
    action = np.array([1,4,5,6,8,9,11])
    dataset = DataProcess(config)
    # res = dataset.item_ranks()
    s = time()
    promoted_items = [100,200,300,400,99,1,25,86,45,10,15,30]
    reward_calculator = RewardCalculator(config,dataset,promoted_items,action)
    res = reward_calculator.average_cnt
    e = time()
    print(e-s)

