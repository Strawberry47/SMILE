# coding: utf-8
import sys
from time import time

from tianshou.data import Batch
from torch.nn import functional

sys.path.append("..")
from SmileEnv import SmileEnv
import numpy as np
import random
from tool import utils
import math
import torch
import torch.nn as nn


# s -> logits
class UserSelector(nn.Module):
    def __init__(self, config,child_num,device):
        super(UserSelector, self).__init__()
        self.config = config
        self.device = device
        self.probs = []
        # self.state_tracker = statetracker
        # self.env = envs  # dummy
        self.log = utils.Log()  # 用于输出日志和时间
        self.episode_len = int(self.config['META']['EPISODE_LENGTH'])
        self.action_dim = int(self.config['META']['ACTION_DIM'])  # 32 embedding维度
        self.reward_dim = int(self.config['META']['REWARD_DIM'])  # 1
        self.state_dim = int(self.config['META']['STATE_DIM'])
        # 加载层次树
        self.output = self.config['ENV']['OUT_PUT']
        self.bc_dim = int(self.config['USERSELECTOR']['TREE_DEPTH'])  # 深度
        self.child_num = child_num

        self.tree_file_path = self.output + 'tree_model_%s_%s_c%d' % (
            self.config['USERSELECTOR']['CLUSTERING_VECTOR_TYPE'].lower(),
            self.config['USERSELECTOR']['CLUSTERING_TYPE'].lower(), self.child_num)
        # top_to_bottom，leaf_id，dataset，child_num，clustering_type
        tree_model = utils.pickle_load(self.tree_file_path)
        # 相当于self.bc_embeddings和self.code2id
        self.top_to_bottom, self.leaf_id = (tree_model['top_to_bottom'], tree_model['leaf_id'])
        # get available nodes
        self.aval_val = self.get_available_nodes()

        policy_num = int((math.pow(self.child_num, self.bc_dim) - 1) / (self.child_num - 1))

        # 定义actor
        actor = nn.Sequential(
            nn.Linear(self.state_dim,self.child_num),
            nn.ReLU()
        )
        # create actors (input state,output action pro)
        self.actors = nn.ModuleList([actor for i in range(policy_num)])

    def get_pro_by_index(self, state, policy_index):
        # initial
        # actions = self.actors[policy_index[0].int().clone().detach()](state[0]).unsqueeze_(0)
        actions = []
        for i in range(len(policy_index)):
            p_i = policy_index[i].int().clone().detach()
            s_i = state[i]
            action = self.actors[p_i](s_i).unsqueeze(0)
            actions.append(action)
        actions = torch.cat(actions,dim=0)
        return actions

    def forward(self,state,training=False):
        # input batchsize states, output batchsize pro and actions
        self.batch_size = len(state)
        self.aval_list = torch.tensor(np.tile(np.expand_dims(self.aval_val, 1), [1, self.batch_size, 1])).to(
            self.device)
        self.pre_shift = torch.tensor(np.zeros(shape=[self.batch_size])).to(self.device)
        self.pre_mul_choice = torch.tensor(np.zeros(shape=[self.batch_size])).to(self.device)
        self.index_in_leaf = torch.tensor(np.zeros(shape=[self.batch_size])).to(self.device)
        self.aval_list_t = self.aval_list  # 把aval_val扩展成(10,train_batch_size,111)
        # s1 = time()
        for i in range(self.bc_dim):  # 有几层就要采样几次
            # record the policy index
            # pre_mul_choice存储每个env选择的action
            self.forward_index = self.node_num_before_depth_i(i) + self.child_num * self.pre_shift + self.pre_mul_choice
            self.pre_shift = self.child_num * self.pre_shift + self.pre_mul_choice
            # generate probs
            self.forward_prob = self.get_pro_by_index(state, self.forward_index)

            # 这里拼接的是 [[0],[1,]...[batchsize]],[0],[0],...[0];在第一个维度拼接[[1,0],[2,0]...[14,0]];tile后，重复十次
            concat_content = [torch.arange(self.batch_size).unsqueeze(1).to(self.device), self.forward_index.unsqueeze(1)]
            tile_content = (torch.cat(concat_content, dim=1)).repeat([self.child_num, 1])
            # 生成0~child_num的list，再增加维度，变成[[0],[1]...[9]];tile后扩展成[[0,0...,0],[1,1...1]..[9,9...9]]数量为batchsize
            # 再进行reshape (batch_size*child_num,1)
            reshape_content = torch.reshape((torch.arange(self.child_num).unsqueeze(1)).repeat([1, self.batch_size]), [-1, 1]).to(self.device)
            overall_concat = [reshape_content, tile_content]
            # 第一列表示孩子数 第二列是batchsize 第三类是forward index (batchsize,10,3)
            self.gather_index = torch.reshape(torch.cat(overall_concat, dim=1),
                                              [self.child_num, self.batch_size, 3]).permute(1, 0, 2)
            # shape: batchsize*child_num; 代表每个batch当前节点(policy)的child的可用leaf
            self.aval_item_num_list = self.gather_nd(self.aval_list, self.gather_index)
            # 将每一行的10个元素加起来(batch_size,1)；相除后 代表行每个元素在该行占得比例
            self.aval_prob = self.aval_item_num_list / torch.sum(self.aval_item_num_list,dim=1,keepdim=True)
            self.mix_prob = torch.clamp(self.forward_prob, min=1e-30, max=1.0)* self.aval_prob
            # produce the selected actions of each batch by probability
            if not training:
                self.pre_mul_choice = torch.squeeze(torch.argmax(input=self.mix_prob, dim=1)).float()
            else:
                self.pre_mul_choice = torch.squeeze(torch.multinomial(input=self.mix_prob, num_samples=1, replacement=False)).float()

            # 处理aval_list,选中的节点要删掉
            # 先把forward_index(节点索引，到了哪个非叶节点)转成one-hot形式（长度为策略网络数目，就是对应上是哪个策略网络）(batchsize,1111)
            forward_one_hot = functional.one_hot(self.forward_index.long(), num_classes=self.node_num_before_depth_i(self.bc_dim))

            # 长度为batchsize的1
            ones_res = np.ones(shape=[self.batch_size])
            # 依次比较pre_mul_choice和节点0~9是否相等，返回True或者false(长度为batchsize)；再转成1或0的形式[[1],[0]..[0]](15,1)
            # 乘上one-hot，就对应上选择的子节点了！ 再把他们删除
            self.aval_list = self.aval_list - torch.cat([(
                forward_one_hot * (torch.eq(self.pre_mul_choice, torch.tensor(ones_res * j,dtype=float).to(self.device)).unsqueeze(1).float())).unsqueeze(0) for j in range(self.child_num)],
                dim=0)
            # index in leaf
            self.index_in_leaf = self.index_in_leaf * self.child_num + self.pre_mul_choice.int()

        # print("SMILE sampling time: %f s" % ((time() - s1)))
        ## update avalable children items at each node \
        self.aval_list_t = self.aval_list
        # index in dataset
        forward_sampled_action = torch.index_select(torch.tensor(self.leaf_id).to(self.device), 0, self.index_in_leaf.int())
        return self.mix_prob, forward_sampled_action

    def get_pro_by_action(self,states,actions):
        # return log_pro
        self.bc_embeddings = torch.tensor(self.top_to_bottom,dtype=float).to(self.device)
        self.cur_action = torch.from_numpy(actions).to(self.device)
        batch_size = len(states)
        ## get policy network outputs
        # (batchsize,depth)，拼接0和当前action的长辈[[0,gf,fa],[0,gf,fa]]; 用cur_action(placeholder)查表bc_embeddings（id_to_code）,返回的是除了最后一列的数据
        self.pre_c = torch.cat((torch.zeros([batch_size, 1]).to(self.device),torch.index_select(self.bc_embeddings, 0, self.cur_action)[:, 0:-1]),dim=1).int()
        self.pre_con = torch.zeros([batch_size,1]).to(self.device)
        self.index = []


        # 循环深度次
        for i in range(self.bc_dim): # 后面查找W表需要
            # 第i层前的node数量 + 孩子数 * pre_con是Variable 一开始初始化为0了 + pre_c(拼了cur_action)
            self.index.append(self.node_num_before_depth_i(i) + self.child_num * self.pre_con + self.pre_c[:, i:i+1])
            self.pre_con = self.pre_con * self.child_num + self.pre_c[:, i:i+1] # 乘孩子数 + action对应的父节点

        self.index = torch.cat(self.index, dim=1) # batchsize,depth # 后面查找W表需要
        self.pro_outputs = []

        # 这里是输入state pn_outputs存softmax结果
        for i in range(self.bc_dim):
            prob = self.get_pro_by_index(states, self.index[:, i].int())
            self.pro_outputs.append(prob)

        # 当前action的辈分表
        self.a_code = torch.index_select(self.bc_embeddings, 0, self.cur_action).to(self.device)
        # 选择这个action的概率？
        expand_dim = torch.arange(batch_size).unsqueeze(1).to(self.device)
        log_pi = torch.sum(torch.cat([torch.log(torch.clamp(self.gather_nd(self.pro_outputs[i],
                                             torch.cat([expand_dim, self.a_code[:, i:i + 1].int()], dim=1))
                              , min=1e-30, max=1.0)).unsqueeze(1) for i in range(self.bc_dim)], dim=1),dim=1)

        return log_pi


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


    # 每个非叶节点的每个子树中可用的item目数
    def get_available_nodes(self):
        # 传入深度，返回该深度前的所有节点数
        aval_list = np.zeros(shape=[self.child_num, self.node_num_before_depth_i(self.bc_dim)], dtype=int)
        # 返回有孩子的节点
        self.rec_get_aval(aval_list, self.node_num_before_depth_i(self.bc_dim - 1),
                          list(map(lambda x: int(x >= 0), self.leaf_id)))
        # self.log.log('get_available_nodes completed')
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


