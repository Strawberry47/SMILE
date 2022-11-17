# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 14:32
# @Author  : Shiqi Wang
# @FileName: PureMF.py
# get user and item embedding by mf
from time import time

import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import os
from numba import njit, jit
import pandas as pd

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 config,dataProcess):
        super(PureMF, self).__init__()
        trainingData = dataProcess.rating
        self.data = np.array(trainingData)
        # users = self.dataProcess.user2id[trainingData[:0]]
        self.num_users = dataProcess.user_num
        self.num_items = dataProcess.item_num
        self.latent_dim = int(config['META']['ITEM_DIM'])

        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print("use NORMAL distribution initilizer for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())  # 倒置
        return self.f(scores)

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        self.users_emb = self.embedding_user(users)
        self.items_emb = self.embedding_item(items)
        scores = torch.sum(self.users_emb * self.items_emb, dim=1)
        return self.f(scores)

    def cal_loss(self, users, items, ground_truth):
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        y_pred = torch.sum(users_emb * items_emb, dim=1)
        loss_fun = nn.MSELoss(reduction='sum')
        # 必须加.float
        target_loss = loss_fun(y_pred, ground_truth.float())
        reg_loss = (1/2)*(self.users_emb.norm(2).pow(2) +
                          self.items_emb.norm(2).pow(2))/float(len(users))
        loss = target_loss+reg_loss
        return loss

class BPR(BasicModel):
    def __init__(self,
                 config,interaction_df):
        super(BPR, self).__init__()
        self.num_users = interaction_df['userid'].nunique()
        self.num_items = interaction_df['itemid'].nunique()
        self.latent_dim = int(config['META']['ITEM_DIM'])

        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print("use NORMAL distribution initilizer for BPR")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())  # 倒置
        return self.f(scores)

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        self.users_emb = self.embedding_user(users)
        self.items_emb = self.embedding_item(items)
        scores = torch.sum(self.users_emb * self.items_emb, dim=1)
        return self.f(scores)

    def cal_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss



class MFDataset(Dataset):
    def __init__(self, u_id, i_id, rating,device):
        super(MFDataset, self).__init__()
        self.u_id = torch.Tensor(u_id).long().to(device)
        self.i_id = torch.Tensor(i_id).long().to(device)
        self.rating = torch.Tensor(rating).long().to(device)


    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)

class BPRDataset(Dataset):
    def __init__(self, df, device):
        super(BPRDataset, self).__init__()
        self.device = device
        sparseMatrix = csr_matrix((np.ones(len(df)), (df['userid'], df['itemid'])),
                               shape=(df['userid'].max() + 1, df['itemid'].max() + 1),
                               dtype=np.bool).toarray()

        df_negative = np.zeros([len(df), 2])
        self.find_negative(df['userid'].to_numpy(), df['itemid'].to_numpy(), sparseMatrix, df_negative,
                      df['userid'].max())
        df_negative = pd.DataFrame(df_negative, columns=["userid", "neg_itemid"], dtype=int)
        self.pairwise = pd.concat([df[['userid'] + ['itemid']],df_negative['neg_itemid']],axis=1)
        print("ok")


    def __getitem__(self, idx):
        pairwise = self.pairwise

        users = torch.Tensor(pairwise['userid'].to_numpy()).long()
        posItems = torch.Tensor(pairwise['itemid'].to_numpy()).long()
        negItems = torch.Tensor(pairwise['neg_itemid'].to_numpy()).long()

        users = users.to(self.device)
        posItems = posItems.to(self.device)
        negItems = negItems.to(self.device)

        return users[idx], posItems[idx], negItems[idx]


    # @njit
    def find_negative(self,user_ids, item_ids, mat, df_negative, max_item):
        for i in range(len(user_ids)):
            user, item = user_ids[i], item_ids[i]  # 一条一条地取

            neg = item + 1
            while neg <= max_item:
                if neg == 1225:  # 1225 is an absent video_id
                    neg = 1226
                if mat[user, neg]:  # True # 在大矩阵或小矩阵都是有评分的
                    neg += 1
                else:  # 找到了负样本，就退出
                    df_negative[i, 0] = user
                    df_negative[i, 1] = neg
                    break
            else:
                neg = item - 1
                while neg >= 0:
                    if mat[user, neg]:
                        neg -= 1
                    else:
                        df_negative[i, 0] = user
                        df_negative[i, 1] = neg
                        break

    def __len__(self):
        return len(self.rating)