# -*- coding: utf-8 -*-
# @Time    : 2022/5/26 0:09
# @Author  : Shiqi Wang
# @FileName: train.py
import gc
import math
from time import time
from torch.nn import functional
import numpy as np
import torch
from torch import nn
from tianshou.data import (
    Batch,
    to_numpy,
)

from tool import utils


class trainer():
    def __init__(self,child_num,user_selector,state_tracker,envs,optim,config):
        self.user_selector = user_selector
        self.state_tracker = state_tracker
        self.env = envs  # dummy
        self.env_num = len(envs)
        self.optim = optim
        self.config = config
        self.episode_len = int(self.config['META']['EPISODE_LENGTH'])
        # load tree structure
        self.output = self.config['ENV']['OUT_PUT']
        self.bc_dim = int(self.config['USERSELECTOR']['TREE_DEPTH'])  # 深度
        self.child_num = child_num
        self.log = utils.Log()  # 用于输出日志和时间
        self.tree_file_path = self.output + 'tree_model_%s_%s_c%d' % (
            self.config['USERSELECTOR']['CLUSTERING_VECTOR_TYPE'].lower(),
            self.config['USERSELECTOR']['CLUSTERING_TYPE'].lower(), self.child_num)
        # top_to_bottom，leaf_id，dataset，child_num，clustering_type
        tree_model = utils.pickle_load(self.tree_file_path)
        # 相当于self.bc_embeddings和self.code2id
        self.top_to_bottom, self.leaf_id = (tree_model['top_to_bottom'], tree_model['leaf_id'])
        # get available nodes
        self.aval_val = self.get_available_nodes()


    # 每个非叶节点的每个子树中可用的item目数
    def get_available_nodes(self):
        # 传入深度，返回该深度前的所有节点数
        aval_list = np.zeros(shape=[self.child_num, self.node_num_before_depth_i(self.bc_dim)], dtype=int)
        # 返回有孩子的节点
        self.rec_get_aval(aval_list, self.node_num_before_depth_i(self.bc_dim - 1),
                          list(map(lambda x: int(x >= 0), self.leaf_id)))
        self.log.log('get_available_nodes completed')
        return aval_list

    # 又是递归调用；aval_list表示每个非叶节点的可用item数目；start_index表示从第几层的node算；l中0表示
    def rec_get_aval(self, aval_list, start_index, l):
        if len(l) == 1:  # 算到根节点结束
            return
        new_l = []
        for i in range(int(len(l) / self.child_num)):  # 叶节点的父节点个数 100，也就是倒数第二层的节点数
            index = start_index + i  # 最开始是111~211
            for j in range(self.child_num):  # 循环10次
                aval_list[j][index] = l[self.child_num * i + j]  # 等式右边是1或0
            new_l.append(np.sum(aval_list[:, index]))
        self.rec_get_aval(aval_list, int(start_index / self.child_num), new_l)

    # 返回在i层前的节点数量
    def node_num_before_depth_i(self, i):
        return int((math.pow(self.child_num, i) - 1) / (self.child_num - 1))


    def train(self,device):
        # self.train_batch_size = int(self.config['META']['TRAINNUM'])
        self.device = device
        self.train_batch_size = self.env_num
        self.buffer = self.collect()


    def collect(self):
        # self.data 初始化；state初始化
        self.collector_reset()
        ready_env_ids = np.arange(self.env_num)

        buffer = [[[]], [[]], [[]]] # state, action, reward,
        step_count = 0

        while True:
            # 传入一个batch的state，生成一个batch的action
            sampled_action = self.generate_action(self.data.obs)
            act = to_numpy(sampled_action)
            self.data.update(act=act)
            obs_next, rew, done, info = self.env.step(
                act, ready_env_ids)  # type: ignore
            # 更新数据 .update方法是tianshou库中的Batch类自带的
            self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)
            # 传入action和reward，得到下一个state
            self.data.update(self.state_tracker.build_state(
                obs_next=self.data.obs_next,
                rew=self.data.rew,
                done=self.data.done,
                info=self.data.info,
                env_id=ready_env_ids,
            ))

            self.data.obs = self.data.obs_next  # Todo: 注意这里的状态更新
            step_count = step_count+1
            if step_count==self.episode_len:
                break
        return buffer


    def collector_reset(self):
        self.data = Batch(obs={}, act={}, rew={}, done={},
                          obs_next={}, info={})
        # reset env
        self.state_tracker.build_state(env_num=self.env_num, reset=True)
        obs = self.env.reset()  # 返回了env_num个item
        # 构造batchsize个state
        obs = self.state_tracker.build_state(
            obs=obs, env_id=np.arange(self.env_num)).get("obs", obs)
        self.data.obs = obs

    def generate_action(self,state):
        # input batchsize states, output actions
        self.aval_list = torch.tensor(np.tile(np.expand_dims(self.aval_val, 1), [1, self.train_batch_size, 1])).to(
            self.device)
        self.pre_shift = torch.tensor(np.zeros(shape=[self.train_batch_size])).to(self.device)
        self.pre_mul_choice = torch.tensor(np.zeros(shape=[self.train_batch_size])).to(self.device)
        self.index_in_leaf = torch.tensor(np.zeros(shape=[self.train_batch_size])).to(self.device)
        self.aval_list_t = self.aval_list  # 把aval_val扩展成(10,train_batch_size,111)
        s1 = time()
        for i in range(self.bc_dim):  # 有几层就要采样几次
            # record the policy index
            # pre_mul_choice存储每个env选择的action
            self.forward_index = self.node_num_before_depth_i(i) + self.child_num * self.pre_shift + self.pre_mul_choice
            self.pre_shift = self.child_num * self.pre_shift + self.pre_mul_choice
            # generate probs
            self.forward_prob = self.user_selector(state,self.forward_index)

            # 这里拼接的是 [[0],[1,]...[batchsize]],[0],[0],...[0];在第一个维度拼接[[1,0],[2,0]...[14,0]];tile后，重复十次
            concat_content = [torch.arange(self.train_batch_size).unsqueeze(1).to(self.device),self.forward_index.unsqueeze(1)]
            tile_content = (torch.cat(concat_content, dim=1)).repeat([self.child_num, 1])
            # 生成0~child_num的list，再增加维度，变成[[0],[1]...[9]];tile后扩展成[[0,0...,0],[1,1...1]..[9,9...9]]数量为batchsize
            # 再进行reshape (batch_size*child_num,1)
            reshape_content = torch.reshape((torch.arange(self.child_num).unsqueeze(1)).repeat([1, self.train_batch_size]),[-1,1]).to(self.device)
            overall_concat = [reshape_content, tile_content]
            # 第一列表示孩子数 第二列是batchsize 第三类是forward index (batchsize,10,3)
            self.gather_index = torch.reshape(torch.cat(overall_concat, dim=1),
                          [self.child_num, self.train_batch_size, 3]).permute(1, 0, 2)
            # shape: batchsize*child_num; 代表每个batch当前节点(policy)的child的可用leaf
            self.aval_item_num_list = self.gather_nd(self.aval_list, self.gather_index)
            # 将每一行的10个元素加起来(batch_size,1)；相除后 代表行每个元素在该行占得比例
            self.aval_prob = self.aval_item_num_list / torch.sum(self.aval_item_num_list,dim=1,keepdim=True)
            self.mix_prob = torch.clamp(self.forward_prob, min=1e-30, max=1.0)* self.aval_prob
            # produce the selected actions of each batch by probability
            self.pre_mul_choice = torch.squeeze(torch.multinomial(input=self.mix_prob, num_samples=1, replacement=False)).float()

            # 处理aval_list,选中的节点要删掉
            # 先把forward_index(节点索引，到了哪个非叶节点)转成one-hot形式（长度为策略网络数目，就是对应上是哪个策略网络）(batchsize,1111)
            forward_one_hot = functional.one_hot(self.forward_index.long(), num_classes=self.node_num_before_depth_i(self.bc_dim))

            # 长度为batchsize的1
            ones_res = np.ones(shape=[self.train_batch_size])
            # 依次比较pre_mul_choice和节点0~9是否相等，返回True或者false(长度为batchsize)；再转成1或0的形式[[1],[0]..[0]](15,1)
            # 乘上one-hot，就对应上选择的子节点了！ 再把他们删除
            self.aval_list = self.aval_list - torch.cat([(
                forward_one_hot * (torch.eq(self.pre_mul_choice, torch.tensor(ones_res * j,dtype=float).to(self.device)).unsqueeze(1).float())).unsqueeze(0) for j in range(self.child_num)],
                dim=0)
            # index in leaf
            self.index_in_leaf = self.index_in_leaf * self.child_num + self.pre_mul_choice.int()

        # print("SMILE sampling time: %f s" % ((time() - s1)))

        ## update avalable children items at each node 把减过的list赋给updata
        # train中的，tf.assign：把aval_list的值赋给aval_list_t,aval_list_t的值必须是Variable
        self.aval_list_t = self.aval_list
        # index in dataset
        forward_sampled_action = torch.index_select(torch.tensor(self.leaf_id).to(self.device), 0, self.index_in_leaf.int())
        return forward_sampled_action


    def evulate_model(self):
        print("~" * 30, "evaluate", "~" * 30)
        # initialize
        self.is_eval = True
        self.eval_batch_size = int(self.config['META']['TESTNUM'])
        self.aval_eval_list = np.tile(np.expand_dims(self.aval_val, 1), [1, self.eval_batch_size, 1])
        self.pre_shift_eval = torch.tensor(np.zeros(shape=[self.train_batch_size]))
        self.pre_max_choice_eval = torch.tensor(np.zeros(shape=[self.train_batch_size]))
        self.action_index_eval = torch.tensor(np.zeros(shape=[self.train_batch_size]))
        self.aval_eval_list_t = self.aval_eval_list  # (10, eval_batch_size, 111)



    def gather_nd(self,params, indices):
        ''' 4D example params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices returns: tensor shaped [m_1, m_2, m_3, m_4] ND_example params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices returns: tensor shaped [m_1, ..., m_1] '''
        out_shape = indices.shape[:-1]
        indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
        ndim = indices.shape[0]
        indices = indices.long()
        idx = torch.zeros_like(indices[0], device=indices.device).long()
        m = 1
        for i in range(ndim)[::-1]:
            idx += indices[i] * m
            m *= params.size(i)
        out = torch.take(params, idx)
        return out.view(out_shape)