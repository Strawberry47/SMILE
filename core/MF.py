# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 14:32
# @Author  : Shiqi Wang
# @FileName: MF.py
# get user and item embedding by mf
from time import time

import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import os
import pandas as pd
from numba import njit, jit

from core.utils import find_negative


class PureMF(nn.Module):
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
        # print("use NORMAL distribution initilizer for PureMF")

    def getUsersRating(self, users,items=None):
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
        self.users_emb = self.embedding_user(users.long())
        self.items_emb = self.embedding_item(items.long())
        y_pred = torch.sum(self.users_emb * self.items_emb, dim=1)
        loss_fun = nn.MSELoss(reduction='sum')

        target_loss = loss_fun(y_pred, ground_truth.float())
        reg_loss = (1/2)*(self.users_emb.norm(2).pow(2) +
                          self.items_emb.norm(2).pow(2))/float(len(users))
        loss = target_loss+reg_loss
        return loss

class BPR(nn.Module):
    def __init__(self,
                 config,interaction_df,device):
        super(BPR, self).__init__()
        self.num_users = interaction_df['userid'].nunique()
        self.num_items = interaction_df['itemid'].nunique()
        self.latent_dim = int(config['META']['ITEM_DIM'])
        self.device = device

        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        # print("use NORMAL distribution initilizer for BPR")

    def getUsersRating(self, users, items):
        users = torch.Tensor(users).long().to(self.device)
        items = torch.Tensor(items).long().to(self.device)

        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.matmul(users_emb, items_emb.t())  # 倒置

        return self.f(scores)

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        self.users_emb = self.embedding_user(users)
        self.items_emb = self.embedding_item(items)
        scores = torch.sum(self.users_emb * self.items_emb, dim=1)
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_emb.norm(2).pow(2) +
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss

    def cal_loss(self, users, pos, neg):
        weight_decay = 1e-4
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        reg_loss = reg_loss * weight_decay
        total_loss = loss + reg_loss
        return total_loss


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
    def __init__(self, config, df, device):
        super(BPRDataset, self).__init__()
        self.device = device
        self.df = df
        # negative_sample_path = os.path.join(config['ENV']['OUT_PUT'], "negative_sample")
        sparseMatrix = csr_matrix((np.ones(len(df)), (df['userid'], df['itemid'])),
                               shape=(df['userid'].max() + 1, df['itemid'].max() + 1),
                               dtype=np.bool).toarray()

        df_negative = np.zeros([len(df), 2])
        find_negative(df['userid'].to_numpy(), df['itemid'].to_numpy(), sparseMatrix, df_negative,
                      df['itemid'].max())
        df_negative = pd.DataFrame(df_negative, columns=["userid", "neg_itemid"], dtype=int)
        pairwise = pd.concat([df[['userid'] + ['itemid']],df_negative['neg_itemid']],axis=1)

        users = torch.Tensor(pairwise['userid'].to_numpy()).long()
        posItems = torch.Tensor(pairwise['itemid'].to_numpy()).long()
        negItems = torch.Tensor(pairwise['neg_itemid'].to_numpy()).long()

        self.users = users.to(self.device)
        self.posItems = posItems.to(self.device)
        self.negItems = negItems.to(self.device)

    def __getitem__(self, idx):
        return self.users[idx], self.posItems[idx], self.negItems[idx]


    def __len__(self):
        return len(self.df)


