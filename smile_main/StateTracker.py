# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 11:17
# @Author  : Shiqi Wang
# @FileName: StateTracker.py
import configparser
from time import time

import numpy as np
import torch
from torch import nn, Tensor

from ProcessingData.dataProcess import DataProcess

FLOAT = torch.FloatTensor

# 输入action和reward，输出state
class StateTrackerGRU(nn.Module):
    def __init__(self, config, input_dim, hidden_dim, output_dim,
                 user_emb, item_emb,  EPISODE_LENGTH, device,seed=2022,):
        super(StateTrackerGRU, self).__init__()
        torch.manual_seed(seed)
        self.EPISODE_LENGTH = EPISODE_LENGTH
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.dim_model = int(input_dim)
        self.hidden_dim = hidden_dim
        self.device = device
        self.config = config

        user_embedding_file_path = self.config['ENV']['OUT_PUT'] + 'user_embedding_dim%d' % int(self.config['META']['ACTION_DIM'])
        item_embedding_file_path = self.config['ENV']['OUT_PUT'] + 'item_embedding_dim%d' % int(self.config['META']['ACTION_DIM'])
        user_embedding = torch.tensor(np.loadtxt(user_embedding_file_path, dtype=float, delimiter='\t'), dtype=torch.float32)
        item_embedding = torch.tensor(np.loadtxt(item_embedding_file_path, dtype=float, delimiter='\t'), dtype=torch.float32)

        self.user_embedding = nn.Embedding.from_pretrained(user_embedding)
        self.item_embedding = nn.Embedding.from_pretrained(item_embedding)

        # 初始化GRU模型
        self.GRU_layer = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        # GRU初始输入
        self.ffn_item = nn.Linear(item_emb.shape[1], input_dim, device=device)
        self.fnn_gate = nn.Linear(1 + user_emb.shape[1], input_dim, device=device)
        # g_t，输入是reward拼action
        self.sigmoid = nn.Sigmoid()

    # 输入拼起来的action和reward，输出state
    def forward(self,input,last_hidden):
        # input是经过build_state处理过的
        x, hidden = self.GRU_layer(input,last_hidden)
        s_t = self.output_linear(x.squeeze())
        return s_t, hidden

    # 将action与reward拼起来
    def build_state(self, obs=None,
                    env_id=None,
                    obs_next=None,  # user
                    rew=None,
                    done=None,
                    info=None,
                    policy=None,
                    env_num=None,
                    reset=False):

        # 第一次，初始化
        if reset and env_num:
            self.data = torch.zeros(self.EPISODE_LENGTH, env_num, self.dim_model,
                                    device=self.device)  # (EPISODE_LENGTH, Batch/env_num, Dim)
            self.len_data = torch.zeros(env_num, dtype=torch.int64) # 记录每个env的长度
            return

        res = {}

        # 1. initialize the state vectors
        if obs is not None:
            # initialize state, all item embedding sum
            e_0 = self.get_embedding(obs, "item")
            e_i = torch.sum(e_0,dim=1)
            e_i_prime = self.ffn_item(e_i) # e_i' (env_num,32)

            length = 1
            self.len_data[env_id] = length
            self.data[0, env_id, :] = e_i_prime # (31,10,32)

            nowdata = self.data[:length, env_id, :] # (1,10,32)
            # # 这里初始化hidden[batch,hidden_size]
            hidden = torch.zeros(length,length,self.hidden_dim,device=self.device) # (1,1,12)
            s0, self.hidden = self.forward(nowdata,hidden)

            res = {"obs": s0}

        # 2. add action autoregressively
        elif obs_next is not None:
            a_t = self.get_embedding(obs_next, "user") # 需要获取action embedding；这里是从user model获取的哦

            self.len_data[env_id] += 1
            length = int(self.len_data[env_id[0]])

            # turn = obs_next[:, -1]
            # assert all(self.len_data[env_id].numpy() == turn + 1)
            rew_matrix = rew.reshape((-1, 1))
            r_t = self.get_embedding(rew_matrix, "feedback")

            # g_t = self.sigmoid(self.fnn_gate(torch.cat((r_t, a_t, r_t * a_t), -1)))
            g_t = self.sigmoid(self.fnn_gate(torch.cat((r_t, a_t), -1)))
            a_t_prime = g_t * a_t
            self.data[length - 1, env_id, :] = a_t_prime

            s_t, self.hidden = self.forward(self.data[length-1:length, env_id, :],self.hidden)

            res = {"obs_next": s_t}

        return res
        # return {"obs": obs, "env_id": env_id, "obs_next": obs_next, "rew": rew,
        #         "done": done, "info": info, "policy": policy}

    def get_embedding(self, X, type):
        if type == "user":
            X = torch.from_numpy(X.squeeze()).to(self.device)
            return self.user_embedding(X)
        if type == "item":
            X = torch.from_numpy(X.squeeze()).to(self.device)
            return self.item_embedding(X)
        if type == "feedback":
            return FLOAT(X).to(self.device)

if __name__ == '__main__':
   pass
