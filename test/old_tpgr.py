# -*- coding: utf-8 -*-
# @Time    : 2022/5/31 15:32
# @Author  : Shiqi Wang
# @FileName: old_tpgr.py
import sys

from src.SmileEnv import SmileEnv
from tool import utils
sys.path.append("..")
from src import run_time_tools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
import numpy as np
import random
import math
import time
import gc
import os

class TPGR():
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.log = utils.Log()  # 用于输出日志和时间

        self.episode_length = int(self.config['META']['EPISODE_LENGTH']) #
        self.action_dim = int(self.config['META']['ACTION_DIM']) # 80 embedding维度
        self.reward_dim = int(self.config['META']['REWARD_DIM']) # 100
        self.discount_factor = float(self.config['META']['DISCOUNT_FACTOR']) # 0.9 越大代表越看重未来收益
        self.log_step = int(self.config['META']['LOG_STEP']) # 5
        # 训练的时候每个用户采样的episode
        self.sample_episodes_per_batch = int(self.config['TPGR']['SAMPLE_EPISODES_PER_BATCH'])
        # the number of sample users in a training batch 每一个batch选的用户数
        self.sample_users_per_batch = int(self.config['TPGR']['SAMPLE_USERS_PER_BATCH']) # 10
        self.learning_rate = float(self.config['TPGR']['LEARNING_RATE']) # 0.001
        self.l2_factor = float(self.config['TPGR']['L2_FACTOR']) # 1e-5

        #to control the smooth of the possibility distribution of the policy 控制策略梯度的平滑
        self.entropy_factor = float(self.config['TPGR']['ENTROPY_FACTOR']) # 5.0
        self.child_num = int(self.config['TPGR']['CHILD_NUM']) # 10
        # batch size of episodes for evaluation 用于评估的批大小
        self.eval_batch_size = int(self.config['TPGR']['EVAL_BATCH_SIZE']) #
        self.train_batch_size = self.sample_episodes_per_batch * self.sample_users_per_batch
        # '../results/BPR-movielens/tpgr/'
        self.output = self.config['ENV']['OUT_PUT']
        self.result_file_path = self.output + 'final_result/' + time.strftime('%Y%m%d%H%M%S') + '_' + self.config['ENV']['ALPHA']
        # 这里存的是rnn参数 '../results/BPR-movielens/tpgr/rnn_model_v1s30' 用于之后使用SRU，更新state
        self.rnn_file_path = self.output + 'rnn_model_%s' % (self.config['TPGR']['RNN_MODEL_VS'])
        # indicating whether loading existing model
        self.load_model = self.config['TPGR']['LOAD_MODEL'] == 'T'
        # '../results/BPR-movielens/tpgr/model/tpgr_model_v1s0' 这个是这部分会存储的
        self.load_model_path = self.output + 'model/tpgr_model_%s' % (self.config['TPGR']['MODEL_LOAD_VS'])
        # '../results/BPR-movielens/tpgr/model/tpgr_model_v1'
        self.save_model_path = self.output + 'model/tpgr_model_%s' % (self.config['TPGR']['MODEL_SAVE_VS'].split('s')[0])
        # 存储构建层次聚类树时 里面存有 id_to_code, code_to_id，dataset,child_num，clustering_type
        # '../results/BPR-movielens/tpgr/tree_model_rating_pca_c10_v1'
        self.tree_file_path = self.output + 'tree_model_%s_%s_c%d_%s' % (self.config['TPGR']['CLUSTERING_VECTOR_TYPE'].lower(),
                              self.config['TPGR']['CLUSTERING_TYPE'].lower(), self.child_num, self.config['TPGR']['TREE_VS'])
        #  the hidden units list for each policy network,策略网络是否有隐藏层
        self.hidden_units = [int(item) for item in self.config['TPGR']['HIDDEN_UNITS'].split(',')] if self.config['TPGR']['HIDDEN_UNITS'].lower() != 'none' else []

        self.forward_env = SmileEnv(self.config)
        self.user_num, self.item_num, self.r_matrix, self.dataprocess  = self.forward_env.get_init_data()

        self.boundry_user_id = int(self.forward_env.user_num * 0.8)# 前80%user
        # 测试数据 1189
        self.test_user_num = int(self.user_num/self.eval_batch_size)*self.eval_batch_size-self.boundry_user_id
        self.bc_dim = int(math.ceil(math.log(self.user_num, self.child_num))) # 深度
        # 再次生成多个env对象 943
        # self.env = Env(self.config, self.user_num, self.item_num, self.r_matrix)
        self.env = [SmileEnv(self.config, self.user_num, self.item_num, self.r_matrix, self.dataprocess)
                    for i in range((max(self.train_batch_size,self.eval_batch_size)))]

        # 18
        self.rnn_input_dim = self.action_dim + self.reward_dim
        self.rnn_output_dim = self.rnn_input_dim
        # state + 隐藏层 + 孩子数 目前是[18,10]；会影响后面W的长度；反正unit长度要么2要么3
        self.layer_units = [self.rnn_output_dim] + self.hidden_units + [self.child_num]

        self.is_eval = False
        self.qs_mean_list = [] #q值
        self.storage = []

        self.training_steps = 0
        # 如果要加载存在的模型
        if self.load_model:
            # 找到版本，并且初始训练步数设为版本号
            self.training_steps = int(self.config['TPGR']['MODEL_LOAD_VS'].split('s')[-1])

        # 这里存了上一步层次聚类的结果，包含id_to_code，code_to_id，dataset，child_num，clustering_type
        tree_model = utils.pickle_load(self.tree_file_path)
        # 获取参数 辈分表；树中序号对应的id
        self.top_to_bottom, self.leaf_id = (tree_model['id_to_code'], tree_model['code_to_id'])
        self.aval_val = self.get_aval() #每个非叶节点的每个子树中可用的item数
        self.log.log('making graph')
        self.make_graph()
        self.sess.run(tf.global_variables_initializer())
        self.log.log('graph made')

    def make_graph(self):
        # placeholders 初始化action，会传入每个batch的action
        self.forward_action = tf.placeholder(dtype=tf.int32, shape=[None], name='forward_action')
        self.forward_reward = tf.placeholder(dtype=tf.float32, shape=[None], name='forward_reward')
        self.forward_rnn_state = tf.placeholder(dtype=tf.float32, shape=[2, None, self.rnn_output_dim], name='forward_rnn_state')
        # Q值 通过reward计算出来的
        # 训练的时候更新用
        self.cur_q = tf.placeholder(dtype=tf.float32, shape=[None], name='cur_qs') # Q值
        self.cur_action = tf.placeholder(dtype=tf.int32, shape=[None], name='cur_actions') # 当前的action


        self.action_embeddings = tf.constant(dtype=tf.float32, value=self.forward_env.user_embedding) # PMF的embedding

        # 要通过这个获取item embedding
        targetItem = self.config['ENV']['TARGET_ITEM_LIST'].split(', ')
        self.item_embeddings = self.forward_env.item_embedding
        item_id = [self.dataprocess.item2id[itemname] for itemname in targetItem]
        all_emb = [self.item_embeddings[id] for id in item_id]
        sum_item_emb = np.sum(all_emb,axis=0)

        # id_to_code 辈分表
        self.bc_embeddings = tf.constant(dtype=tf.float32, value=self.top_to_bottom)

        # RNN input
        self.forward_a_emb = tf.nn.embedding_lookup(self.action_embeddings, self.forward_action)
        # 把reward转化为对应维度
        one_hot_reward = tf.one_hot(tf.cast((self.reward_dim*(1.0-self.forward_reward)/2), dtype=tf.int32), depth=self.reward_dim)
        self.forward_ars = tf.concat([self.forward_a_emb, one_hot_reward], axis=1) # RNN输入之一

        # # 初始化pre_rnn_state，这里初始化为0[h,c](2,batch_size,180)
        self.initial_states = tf.stack([tf.zeros([self.train_batch_size, self.rnn_output_dim]),
                                        tf.zeros([self.train_batch_size, self.rnn_output_dim])])

        # 改成item embedding作为输入
        init_s = np.tile(sum_item_emb, (self.train_batch_size, 1))
        self.initial_states = tf.stack([init_s,init_s])

        # 加载rnn模型参数 5个参数
        l = utils.pickle_load(self.rnn_file_path)
        self.rnn, self.rnn_variables = self.create_sru(l)

        # 输入reward+action以及[h,c](初始化是0)，得到state
        self.rnn_state = self.rnn(self.forward_ars, self.forward_rnn_state)
        # state
        self.user_state = tf.concat([self.rnn_state[0]], axis=1)

        # 如果有 就加载
        if self.load_model:
            # 加载模型，为参数赋值 '../results/BPR-movielens/tpgr/model/tpgr_model_v1s0'
            model = utils.pickle_load(self.load_model_path)
            self.W_list = [tf.Variable(model['W_list'][i], dtype=tf.float32) for i in range(len(model['W_list']))]
            self.b_list = [tf.Variable(model['b_list'][i], dtype=tf.float32) for i in range(len(model['b_list']))]
            self.result_file_path = model['result_file_path']
            self.storage = utils.pickle_load(self.result_file_path)
        # 没有以前的模型
        else:
            # self.layer_units = [self.rnn_output_dim] + self.hidden_units + [self.child_num]；有111个18*10的策略网络
            # init_matrix：传入shape，从服从指定正态分布的序列”中随机取出指定个数的值 [1111,180,10] 第一维表示策略网络数量，二维表示state维度,三维代表child_num
            self.W_list = [tf.Variable(self.init_matrix(shape=[self.node_num_before_depth_i(self.bc_dim),
                                                               self.layer_units[i], self.layer_units[i + 1]]))
                           for i in range(len(self.layer_units) - 1)] # 没有隐藏层就是循环1次
            self.b_list = [tf.Variable(self.init_matrix(shape=[self.node_num_before_depth_i(self.bc_dim),
                                                               self.layer_units[i + 1]]))
                           for i in range(len(self.layer_units) - 1)]  # (111,10) 最后输出10个结果的softmax 代表选每个子节点的概率

        # map hidden state to action
        ## variables 树节点中索引对应在数据集中的id
        self.leaf2id = tf.constant(value=self.leaf_id, dtype=tf.int32)
        # tile() 函数,就是将原矩阵横向、纵向地复制；np.expand_dims:用于扩展数组的形状;
        # 把aval_val扩展成(10,train_batch_size,111);可用子树
        self.aval_list = tf.Variable(np.tile(np.expand_dims(self.aval_val, 1), [1, self.train_batch_size, 1]), dtype=tf.float32)
        # evulate
        self.aval_eval_list = tf.Variable(np.tile(np.expand_dims(self.aval_val, 1), [1, self.eval_batch_size, 1]), dtype=tf.float32)
        self.eval_probs = []

        ## constant
        self.pre_shift = tf.constant(value=np.zeros(shape=[self.train_batch_size]), dtype=tf.int32)
        self.pre_mul_choice = tf.constant(value=np.zeros(shape=[self.train_batch_size]), dtype=tf.int32) # 返回一个0~c的值
        self.action_index = tf.constant(value=np.zeros(shape=[self.train_batch_size]), dtype=tf.int32) # action的index，在树中
        self.pre_shift_eval = tf.constant(value=np.zeros(shape=[self.eval_batch_size]), dtype=tf.int32)
        self.pre_max_choice_eval = tf.constant(value=np.zeros(shape=[self.eval_batch_size]), dtype=tf.int32)# 选择最大的概率
        self.action_index_eval = tf.constant(value=np.zeros(shape=[self.eval_batch_size]), dtype=tf.int32)# evaluate中的action index

        self.aval_list_t = self.aval_list # 把aval_val扩展成(10,train_batch_size,111)
        self.aval_eval_list_t = self.aval_eval_list # (10, eval_batch_size, 111)

        ## 输入state，往下走d次，直到选到了叶节点 action_index
        ## 这是训练过程的代码
        s1= time.time()
        for i in range(self.bc_dim): #有几层就要采样几次
            # 记录节点索引 从第一层开始(0) 然后第二层(1+10*[0,0,0..]+上一次选择结果) 也就是到哪个策略网络了
            self.forward_index = self.node_num_before_depth_i(i) + self.child_num * self.pre_shift + tf.cast(self.pre_mul_choice, tf.int32)
            if i == 0: # 初始化state就是user_state
                h = self.user_state
            else: # 不是初始化了
                h = tf.expand_dims(self.user_state, axis=1)
            for k in range(len(self.W_list)): # 如果没有隐藏层就直接套mlp+softmax
                if k == (len(self.W_list) - 1): # 没有隐藏层，直接softmax
                    # for speeding up, do not use embedding_lookup when i==0.
                    if i == 0:
                        # 根节点开始第一轮选择，不用找W参数，直接用根节点的策略网络就好啦 W_list[0][0]；h是state哦！ 111个18*10的策略网络
                        self.forward_prob = tf.nn.softmax(tf.matmul(h, self.W_list[k][0]) + self.b_list[k][0], axis=1)
                    else: # 通过非叶节点寻找对应的策略网络(batchsize个长度为10的向量)；输入state，每个策略网络的输出，代表选择哪一个孩子节点
                        # matmul表示矩阵乘法
                        self.forward_prob = tf.nn.softmax(tf.squeeze(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.forward_index)) +
                                                                 tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.forward_index), axis=1)), axis=1)
                else: # 有隐藏层，就先经过隐藏层 relu
                    if i == 0:
                        h = tf.nn.relu(tf.matmul(h, self.W_list[k][0]) + self.b_list[k][0])
                    else:
                        h = tf.nn.relu(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.forward_index)) +
                                   tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.forward_index), axis=1))

            self.pre_shift = self.child_num * self.pre_shift + tf.cast(self.pre_mul_choice, tf.int32) # 加上了每个batch选择的是哪个子节点
            self.aval_item_num_sum = 0

            # 这里拼接的是 [[0],[1,]...[batchsize]],[0],[0],...[0];在第一个维度拼接[[1,0],[2,0]...[14,0]];tile后，重复十次
            concat_content = [tf.expand_dims(tf.range(self.train_batch_size), axis=1),tf.expand_dims(self.forward_index, axis=1)]
            tile_content = tf.tile(tf.concat(concat_content, axis=1), [self.child_num, 1]) #(300,2)
            # 生成0~child_num的list，再增加维度，变成[[0],[1]...[9]];tile后扩展成[[0,0...,0],[1,1...1]..[9,9...9]]数量为batchsize
            # 再进行reshape (batch_size*child_num,1)
            reshape_content = tf.reshape(tf.tile(tf.expand_dims(tf.range(self.child_num), 1), [1, self.train_batch_size]), [-1, 1])
            overall_concat = [reshape_content, tile_content]
            # 0维和1维换了 变成了[[ [0,0,0],[1,0,0]...[9,0,0]], [[0,1,0]...[9,1,0]], [[0,14,0]..[9,14,0]]]
            # 第一列表示孩子数 第二列是batchsize 第三类是forwardindex (batchsize,10,3)
            self.gather_index = tf.transpose(tf.reshape(tf.concat(overall_concat, axis=1),[self.child_num, self.train_batch_size, 3]), [1, 0, 2])

            # 可用item .aval_list：把aval_val扩展成(10,15,111)；(batchsize,10) 每个batch可用的孩子？
            self.aval_item_num_list = tf.gather_nd(self.aval_list, self.gather_index) #(batchsize,10)
            # reducesun是将每一行的10个元素加起来(batch_size,1)；相除后 代表行每个元素在该行占得比例
            self.aval_prob = self.aval_item_num_list / tf.reduce_sum(self.aval_item_num_list, axis=1, keep_dims=True)
            # 将一个张量的值限制在给定的最小值和最大值之间；
            # softmax结果；这里是把概率限制在一个范围内，再乘aval_prob（比例）
            self.mix_prob = tf.clip_by_value(self.forward_prob, clip_value_min=1e-30, clip_value_max=1.0) * self.aval_prob
            # self.mix_prob = self.forward_prob * (1.0 - tf.cast(tf.equal(self.aval_prob, 0.0), tf.float32))
            # 采样概率
            self.real_prob_logit = tf.log(self.mix_prob / tf.reduce_sum(self.mix_prob, axis=1, keep_dims=True))
            # tf.multinomial：从multinomial分布中采样，样本个数是num_samples，每个样本被采样的概率由real_prob_logit给出，结果是array([[0]])的形式
            # 每个batch选了哪个子节点 长度为batchsize # 返回一个0~c的值 按概率选择
            self.pre_mul_choice = tf.cast(tf.squeeze(tf.multinomial(logits=self.real_prob_logit, num_samples=1)), tf.float32)

            # 处理aval_list,选中的节点要删掉
            # 先把forward_index(节点索引，到了哪个非叶节点)转成one-hot形式（长度为策略网络数目，就是对应上是哪个策略网络）(batchsize,1111)
            forward_one_hot = tf.one_hot(indices=self.forward_index,depth=self.node_num_before_depth_i(self.bc_dim))
            # 长度为batchsize的1
            ones_res = np.ones(shape=[self.train_batch_size])
            # 依次比较pre_mul_choice和节点0~9是否相等，返回True或者false(长度为batchsize)；再转成1或0的形式[[1],[0]..[0]](15,1)
            # 乘上one-hot，就对应上选择的子节点了！ 再把他们删除
            self.aval_list = self.aval_list - tf.concat([tf.expand_dims(forward_one_hot * tf.expand_dims(tf.cast(tf.equal(self.pre_mul_choice, tf.constant(ones_res * j,
                                                                        dtype=tf.float32)), tf.float32), axis=1),axis=0) for j in range(self.child_num)], axis=0)

            # 记录node_index; action_index一开始初始化为0，采样d次（每次从1~c中选择），这里是计算每采样一次，index到哪里了
            # 因为采样d次，所以最后记录的一定是action的index(树中)；后面会用到id_to_code
            self.action_index = self.action_index * self.child_num + tf.cast(self.pre_mul_choice, tf.int32)

        print("depth=3 TPGR sampling time: %f s" % ((time.time() - s1)))

        ## 前面是训练过程 这里是评价过程，区别就是这里选的是最大概率（前面是按照概率选择）
        ### for evaluation, using maximum 还存了self.eval_probs 每次softmax的概率

        for i in range(self.bc_dim):
            # pre_max_choice_eval变了
            self.forward_index_eval = self.node_num_before_depth_i(i) + self.child_num * self.pre_shift_eval + tf.cast(self.pre_max_choice_eval, tf.int32)
            if i == 0:
                h = self.user_state # SRU输出
            else:
                h = tf.expand_dims(self.user_state, axis=1)
            for k in range(len(self.W_list)):
                if k == (len(self.W_list) - 1): # 没有隐藏层，直接计算概率；否则要先把state放入隐藏层处理
                    if i == 0: # 第一个策略网络，直接取第一个参数列表就好了
                        self.forward_prob_eval = tf.nn.softmax(tf.matmul(h, self.W_list[k][0]) + self.b_list[k][0], axis=1)
                    else: # 因为有很多个策略网络，第二轮就不知道下一个是谁了，所以首先要查那一个的参数设置
                        self.forward_prob_eval = tf.nn.softmax(tf.squeeze(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.forward_index_eval)) +
                                                                      tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.forward_index_eval), axis=1)), axis=1)

                else:
                    if i == 0:
                        # 有hidden_layer，就先用relu激活函数处理state；等激活函数过完之后就可以进行选择了
                        h = tf.nn.relu(tf.matmul(h, self.W_list[k][0]) + self.b_list[k][0])
                    else:
                        h = tf.nn.relu(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.forward_index_eval)) +
                                   tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.forward_index_eval), axis=1))
            # 把每次输出的概率存起来（都是tensor）是不是后面映射为action需要啊
            self.eval_probs.append(self.forward_prob_eval)
            # 这里重新赋值pre_shift_eval
            self.pre_shift_eval = self.child_num * self.pre_shift_eval + tf.cast(self.pre_max_choice_eval, tf.int32)
            # index # 第一列表示孩子数 第二列是batchsize 第三类是forwardindex (batchsize,10,3)
            self.gather_index_eval = tf.transpose(tf.reshape(tf.concat([tf.reshape(tf.tile(tf.expand_dims(tf.range(self.child_num), 1), [1, self.eval_batch_size]), [-1, 1]),
                                     tf.tile(tf.concat([tf.expand_dims(tf.range(self.eval_batch_size), axis=1), tf.expand_dims(self.forward_index_eval, axis=1)], axis=1),
                                     [self.child_num, 1])], axis=1), [self.child_num, self.eval_batch_size, 3]), [1, 0, 2])
            # 按照 gather_index_eval的格式从aval_eval_list(那个11*10的数组，表示叶节点中可选item)中抽取切片
            self.aval_item_num_eval_list = tf.gather_nd(self.aval_eval_list, self.gather_index_eval)
            # 计算aval_item_num_eval_list占比
            self.aval_prob_eval = self.aval_item_num_eval_list / tf.reduce_sum(self.aval_item_num_eval_list, axis=1, keep_dims=True)
            #将值缩到一个范围内
            self.mix_prob_eval = tf.clip_by_value(self.forward_prob_eval, clip_value_min=1e-30, clip_value_max=1.0) * self.aval_prob_eval
            # 计算占比
            self.real_prob_logit_eval = self.mix_prob_eval / tf.reduce_sum(self.mix_prob_eval, axis=1, keep_dims=True)
            # 这里直接取的最大概率，前面是按照概率采样；将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。删掉维度为1
            # self.pre_max_choice_eval = tf.cast(tf.squeeze(tf.argmax(input=self.real_prob_logit_eval)), tf.float32)
            self.pre_max_choice_eval = tf.cast(tf.squeeze(tf.argmax(input=self.real_prob_logit_eval, axis=1)),tf.float32)
            # 处理了aval_eval_list，也就是 每个非叶节点的每个子树中可用的item数
            self.aval_eval_list = self.aval_eval_list - tf.concat([tf.expand_dims(tf.one_hot(indices=self.forward_index_eval, depth=self.node_num_before_depth_i(self.bc_dim))
                                                                                  * tf.expand_dims(tf.cast(tf.equal(self.pre_max_choice_eval, tf.constant(np.ones(shape=[self.eval_batch_size]) * j,
                                                                                   dtype=tf.float32)), tf.float32), axis=1), axis=0) for j in range(self.child_num)], axis=0)
            self.action_index_eval = self.action_index_eval * self.child_num + tf.cast(self.pre_max_choice_eval, tf.int32)


        # 训练和评估过程的采样action结束；把每层选择的概率拼起来，(batchsize,10*d)
        self.eval_probs = tf.concat(self.eval_probs, axis=1)
        ## update avalable children items at each node 把减过的list赋给updata
        # train中的，tf.assign：把aval_list的值赋给aval_list_t,aval_list_t的值必须是Variable
        self.update_aval_list = tf.assign(self.aval_list_t, self.aval_list)



        # evaluate中的；右边是减过的list
        self.update_aval_eval_list = tf.assign(self.aval_eval_list_t, self.aval_eval_list)

        # assign avalable children items at each node
        # 10,batchsize,1111
        self.aval_eval_list_v = tf.placeholder(dtype=tf.float32, shape=self.aval_eval_list_t.get_shape())
        # 把后面的值赋给前面的
        self.assign_aval_eval_list = tf.assign(self.aval_eval_list_t, self.aval_eval_list_v)
        # 同样的操作 train
        self.aval_list_v = tf.placeholder(dtype=tf.float32, shape=self.aval_list_t.get_shape())
        self.assign_aval_list = tf.assign(self.aval_list_t, self.aval_list_v)
        ## get action 得到选择的user在数据集中的id
        # train的时候记录了action_index，也就是最后选的那个action，但是记录的是在树中的index，这里是转换成数据集中的索引
        # train中会先运行这部分代码
        self.forward_sampled_action = tf.nn.embedding_lookup(self.leaf_id, self.action_index)
        # 选的用户对应的id
        self.forward_sampled_action_eval = tf.nn.embedding_lookup(self.leaf_id, self.action_index_eval)

        ## get policy network outputs
        # (batchsize,depth)，拼接0和当前action的长辈[[0,gf,fa],[0,gf,fa]]; 用cur_action(placeholder)查表bc_embeddings（id_to_code）,返回的是除了最后一列的数据
        self.pre_c = tf.cast(tf.concat([tf.zeros(shape=[self.train_batch_size, 1]),
                                        tf.nn.embedding_lookup(self.bc_embeddings, self.cur_action)[:, 0:-1]], axis=1), dtype=tf.int32)
        # 要训练的参数
        self.pre_con = tf.Variable(tf.zeros(shape=[self.train_batch_size, 1], dtype=tf.int32))
        self.index = []

        # 循环深度次
        for i in range(self.bc_dim): # 后面查找W表需要
            # 第i层前的node数量 + 孩子数 * pre_con是Variable 一开始初始化为0了 + pre_c(拼了cur_action)
            self.index.append(self.node_num_before_depth_i(i) + self.child_num * self.pre_con + self.pre_c[:, i:i+1])
            self.pre_con = self.pre_con * self.child_num + self.pre_c[:, i:i+1] # 乘孩子数 + action对应的父节点

        self.index = tf.concat(self.index, axis=1)  # batchsize,depth # 后面查找W表需要
        self.pn_outputs = []


        # 这里是输入state pn_outputs存softmax结果
        for i in range(self.bc_dim):
            h = tf.expand_dims(self.user_state, axis=1) # RL的输入
            for k in range(len(self.W_list)):
                if k == (len(self.W_list) - 1): # 没有隐藏层；输入h，利用self.index查表 记录输出 (64000,10) 后面run的时候好像是(2,64000)
                    self.pn_outputs.append(tf.nn.softmax(tf.squeeze(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.index[:, i])) +
                                                                    tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.index[:, i]), axis=1))))
                else: # 有隐藏层，先处理state
                    h = tf.nn.relu(tf.matmul(h, tf.nn.embedding_lookup(self.W_list[k], self.index[:, i])) +
                                   tf.expand_dims(tf.nn.embedding_lookup(self.b_list[k], self.index[:, i]), axis=1))


        # b_list + rnn_variables中的b
        self.bias_variables = self.b_list + self.rnn_variables[3:] # 就是把策略网络的参数和rnn里的参数拼起来
        self.weight_variables = self.W_list + self.rnn_variables[:3] # 后面更新
        # 训练的mse=平均 根号下(pn_outputs-0.1) 后续强化学习更新公式涉及到了 乘entropy_factor
        self.train_mse = tf.reduce_mean(tf.square(tf.concat(self.pn_outputs, axis=1) - 1.0/self.child_num), axis=1)
        self.a_code = tf.nn.embedding_lookup(self.bc_embeddings, self.cur_action) #返回当前action对应的辈分

        # 选择这个action的概率？
        expand_dim = tf.expand_dims(tf.range(self.train_batch_size), axis=1) # 0~batchsize，二维[[0],[1],[2]....]
        # i_node = tf.cast(self.a_code[:, i:i + 1], tf.int32) # 从左列返回，然后返回右列;先返回爸爸辈分，再返回儿子的
        # concat_res = tf.concat([expand_dim,i_node], axis=1) # 把上面两个拼接起来
        # gather_res = tf.gather_nd(self.pn_outputs[i], concat_res) #在多维上进行索引；后者是indice，根据后面的选出前面的
        # clip_res = tf.clip_by_value(gather_res, clip_value_min=1e-30, clip_value_max=1.0) # 控制gather里的值
        # # 对控制后的gather(输出概率)中的值，取对数(想到了策略梯度下降的式子)，扩展维度
        # self.log_pi = tf.reduce_sum(tf.concat([tf.expand_dims(tf.log(clip_res), axis=1)for i in range(self.bc_dim)], axis=1), axis=1)
        # 这里就是根据路径算采样到每个cur_action的概率 shape(batchsize,)  pn_outputs记录了softmax输出，会根据父节点序号以及子节点序号索引选择的概率
        self.log_pi = tf.reduce_sum(tf.concat([tf.expand_dims(tf.log(tf.clip_by_value(tf.gather_nd(self.pn_outputs[i],
                                   tf.concat([expand_dim, tf.cast(self.a_code[:, i:i + 1], tf.int32)], axis=1))
                                 , clip_value_min=1e-30, clip_value_max=1.0)), axis=1) for i in range(self.bc_dim)], axis=1),axis=1)
        # run的时候大小为(batchsize,) 就是记录了选择每个action的概率
        # a_code是cur_action对应的辈分
        self.negative_likelyhood = -self.log_pi
        # 这里的w和b是拼了rnn参数的  实现一个列表的元素的相加
        self.l2_norm = tf.add_n([tf.nn.l2_loss(item) for item in (self.weight_variables + self.bias_variables)])
        # 这里是策略梯度下降的公式吧；cur_q是Q值(reward算了discount，并标准化)；entropy_factor是控制策略下降平滑的参数；这个地方就是损失函数，加了L2正则化
        self.weighted_negative_likelyhood_with_l2_norm = self.negative_likelyhood * self.cur_q + self.entropy_factor * self.train_mse + self.l2_factor * self.l2_norm
        # 选择优化器了
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.weighted_negative_likelyhood_with_l2_norm)

    #  每个非叶节点的每个子树中可用的item目数
    # 非子树中的非叶节点有多少item
    def get_aval(self):
        # 传入深度，返回该深度前的所有节点数 (10,111) 表示第3层前面还有111个非叶节点（其实也是策略网络的个数）
        aval_list = np.zeros(shape=[self.child_num, self.node_num_before_depth_i(self.bc_dim)], dtype=int)
        # map表示映射，前面是函数；lambda也是匿名函数，表示x是非负就返回1，否则返回0；因为在初始化code_to_id时，长度是大于等于总用户数的
        self.rec_get_aval(aval_list, self.node_num_before_depth_i(self.bc_dim-1), list(map(lambda x: int(x >= 0), self.leaf_id)))
        self.log.log('get_aval completed')
        return aval_list

    # 又是递归调用；aval_list表示每个非叶节点的可用item数目；start_index表示从第几层的node算；l中0表示
    def rec_get_aval(self, aval_list, start_index, l):
        if len(l) == 1: # 算到根节点结束
            return
        new_l = []
        for i in range(int(len(l)/self.child_num)): # 叶节点的父节点个数 100，也就是倒数第二层的节点数
            index = start_index + i # 最开始是111~211
            for j in range(self.child_num): # 循环10次
                aval_list[j][index] = l[self.child_num*i+j] # 等式右边是1或0
            new_l.append(np.sum(aval_list[: ,index]))
        self.rec_get_aval(aval_list, int(start_index/self.child_num), new_l)

    # 返回在第i层前面的节点数量
    def node_num_before_depth_i(self, i): # 传入的i是深度
        return int((math.pow(self.child_num, i) - 1) / (self.child_num - 1))

    # 返回欧氏距离
    def dis(self, a, b):
        return np.power(np.sum(np.square(a-b)), 0.5)
    # 服从指定正态分布的序列”中随机取出指定个数的值
    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    # TPGR中的创建SRU，直接加载五个参数，返回两个值
    def create_sru(self, l):
        Wf = tf.constant(l[0])
        bf = tf.constant(l[3])

        Wr = tf.constant(l[1])
        br = tf.constant(l[4])

        U = tf.constant(l[2])

        sru_variables = [Wf, Wr, U, bf, br]

        def unit(x, h_c):
            pre_h, pre_c = tf.unstack(h_c)

            # forget gate
            f = tf.sigmoid(tf.matmul(x, Wf) + bf)
            # reset gate
            r = tf.sigmoid(tf.matmul(x, Wr) + br)
            # memory cell
            c = f * pre_c + (1 - f) * tf.matmul(x, U)
            # hidden state
            h = r * tf.nn.tanh(c) + (1 - r) * x

            return tf.stack([h, c])

        return unit, sru_variables

    def standardization(self, q_matrix):
        q_matrix = q_matrix.astype(float)
        q_matrix -= np.mean(q_matrix)
        std = np.std(q_matrix)
        if std == 0.0:
            return q_matrix
        q_matrix /= std
        return q_matrix

    # evaluate中调用的；train也调用
    def update_avalable_items(self, sampled_users):
        sampled_codes = self.top_to_bottom[sampled_users] # 根据userid确定树中辈分表
        if self.is_eval: # 扩展维度 10,batchsize,111
            aval_val_tmp = np.tile(np.expand_dims(self.aval_val, axis=1), [1, self.eval_batch_size, 1])
        else: # expand_dims后变成(10, 1, 11)
            aval_val_tmp = np.tile(np.expand_dims(self.aval_val, axis=1), [1, self.train_batch_size, 1])
        for i in range(len(sampled_codes)):
            code = sampled_codes[i]
            index = 0
            for c in code: # 辈分
                c = int(c)
                aval_val_tmp[c][i][index] -= 1 # 减去选择的user；i代表选择的user；index代表策略网络
                index = self.child_num * index + 1 + c #
        if self.is_eval: # tf.assign(self.aval_eval_list_t, self.aval_eval_list_v)
            self.sess.run(self.assign_aval_eval_list, feed_dict={self.aval_eval_list_v: aval_val_tmp})
        else:
            self.sess.run(self.assign_aval_list, feed_dict={self.aval_list_v: aval_val_tmp})
        del aval_val_tmp
        gc.collect() # 清内存

    # to start with, the hidden state is all zero, for a cold-start user, here the first action is not given by the policy but random or popularity
    # 一开始的hidden state为0，对于冷启动用户，推荐的第一个物品不是根据策略网络来的，而是随机或者流行度
    # 这里是随机选用户,每个batch都选一个action，返回reward；作为初始化的结果
    def _get_initial_ars(self,batch_size=-1): # 传入的batchsize
        result = [[[]], [[]],[[]]]
        if batch_size == -1:
            batch_size = self.train_batch_size
        for i in range(batch_size): # 循环train_batch_size次
            user_id = random.randint(0, self.user_num - 1) # 随机采样user_id
            recnum,reward = self.env[i].get_reward_tpgr(user_id) # 得到reward
            result[0][0].append(user_id) # 把结果存入result action
            result[1][0].append(reward) # reward
            result[2][0].append(recnum)
        return result

    # 训练过程
    def train(self):
        # initialize
        for i in range(self.sample_users_per_batch):
            for j in range(self.sample_episodes_per_batch):
                self.env[i*self.sample_episodes_per_batch+j].reset()

        ars = self._get_initial_ars() # 先采样batchsize 获取reward 0
        self.update_avalable_items(ars[0][0]) # 传入选择的user
        rnn_state = self.sess.run(self.initial_states)

        step_count = 0

        action_list = []

        # sample actions according to the current policy
        while True:
            feed_dict = {self.forward_action: ars[0][step_count], # 传入刚刚采样的一堆action
                         self.forward_reward: ars[1][step_count],
                         self.forward_rnn_state: rnn_state}
            # 更新可用item，采样action
            # update avalable items and sample actions in a run since different multinomial sampling would lead to different result if splitted
            run_list = [self.forward_sampled_action, self.rnn_state, self.update_aval_list]
            # forward_sampled_action是输入state，树形结构输出的action
            sampled_action, rnn_state, _ = self.sess.run(run_list, feed_dict)
            action_list.append(sampled_action)

            ars[0].append([]) # 要开始存reward了
            ars[1].append([])
            ars[2].append([])
            step_count += 1
            # 获取树形结构采样得到的action的reward，一直采样episode长度
            for j in range(self.train_batch_size):
                recnum,reward = self.env[j].get_reward_tpgr(sampled_action[j])
                ars[0][-1].append(sampled_action[j])
                ars[1][-1].append(reward)
                ars[2][-1].append(recnum)

            # episode长度到了 直接break
            if step_count==self.episode_length:
                break

        # standardize the q-values user-wisely
        # 进行一轮采样后，开始训练
        qs = np.array(ars[1])[1:] # 第一个不算（初始化是随机的值），取出一个episode的值；episode*batchsize
        c_reward = np.zeros([len(qs[0])]) # batchsize长度，用来算累计值的
        # 算Q值 折扣值 每一个batch重复算episode次；每个batchsize对应一个Q值，代表在t时刻的累积收益
        for i in reversed(range(len(qs))): #
            c_reward = self.discount_factor * c_reward + qs[i]
            qs[i] = c_reward  # 倒着算的，最后一个不变的
        # Q值平均 这是要输出的
        self.qs_mean_list.append(np.mean(qs))
        for i in range(self.sample_users_per_batch): # 进行标准化
            qs[:, i * self.sample_episodes_per_batch: (i + 1) * self.sample_episodes_per_batch] = \
                self.standardization(qs[:, i * self.sample_episodes_per_batch: (i + 1) * self.sample_episodes_per_batch])

        # 初始化state
        rnn_state = self.sess.run(self.initial_states)
        # update the policy utilizing the REINFORCE algorithm
        # 这里才涉及到更新，前面是获取action；所以episode不应该设置太长了吧
        # 使用一轮episode的结果更新强化学习中的参数列表
        for i in range(step_count): #一个一个更新哦
            feed_dict = {self.forward_action: ars[0][i], # i时刻的action
                         self.forward_reward: ars[1][i], # 要构成初始state的输入呀
                         self.forward_rnn_state: rnn_state,
                         self.cur_action: ars[0][i + 1], # i+1时刻的action
                         self.cur_q: qs[i]} # Q值
            _, rnn_state = self.sess.run([self.train_op, self.rnn_state], feed_dict=feed_dict)

        del ars
        gc.collect()

        self.training_steps += 1

        if self.training_steps % self.log_step == 0:
        # if self.training_steps % 10 == 0: # 训练多少次输出一次
            print('Q值平均: %.5f'%np.mean(self.qs_mean_list))
            self.qs_mean_list = []
            # self.evaluate()


    def _eva_initial_ars(self,batch_size=-1): # 传入的batchsize
        result = [[[]], [[]],[[]]]
        if batch_size == -1:
            batch_size = self.train_batch_size
        for i in range(batch_size): # 循环train_batch_size次
            user_id = random.randint(0, self.user_num - 1) # 随机生成user_id
            recnum,reward = self.env[i].get_reward_tpgr(user_id) # 得到reward
            result[0][0].append(user_id) # 把结果存入result action
            result[1][0].append(reward) # reward
            result[2][0].append(recnum)
        return result

    def evaluate(self):
        print("~"*30,"evaluate","~"*30)
        # initialize
        self.is_eval = True
        # eval_step_num = int(math.ceil(self.user_num / self.eval_batch_size)) # 63
        eval_step_num = 1
        for i in range(0, self.eval_batch_size * eval_step_num):
            self.env[i].reset()  # 重置env

        # 第一轮选用户，得到的reward
        ars = self._eva_initial_ars(self.eval_batch_size * eval_step_num)# 随机采样选择用户，获取reward
        # sample an episode for each user
        for s in range(eval_step_num): # eval_step_num = 1
            start = s * self.eval_batch_size
            end = (s + 1) * self.eval_batch_size # [start:end]表示每个batch的action
            self.update_avalable_items(ars[0][0][start:end])  #传入采样的用户，更新aval_list
            rnn_state = np.zeros([2, self.eval_batch_size, self.rnn_output_dim]) # 初始化为0
            step_count = 0
            # stop_flag = False

            while True:
                # ars[0][0]是随机采样的，输出的sampled_action是根据策略网络采样的
                feed_dict = {self.forward_action: ars[0][step_count][start:end], # 传入初始action batchsize
                             self.forward_reward: ars[1][step_count][start:end], # 传入初始reward
                             self.forward_rnn_state: rnn_state} # 初始state 0
                # action在数据集中的id，运行sru的结果[h,c]，每层概率拼接，赋值更新aval_list
                run_list = [self.forward_sampled_action_eval, self.rnn_state, self.eval_probs, self.update_aval_eval_list]
                result_list = self.sess.run(run_list, feed_dict)
               # 传入初始action和reward；树形结构选出的action在数据集中id，rnn运行结果，每层选择概率的
                sampled_action, rnn_state, probs = result_list[0:3] # sample_action代表输入初始action和reward后，树形结构给batch的下一个action
                # ars[0]
                step_count += 1
                if len(ars[0]) == step_count: # 长度一样，一开始就满足
                    ars[0].append([])
                    ars[1].append([])
                    ars[2].append([]) # 存的是RecNum
                for j in range(self.eval_batch_size): # 策略网络采样出的action，算reward
                    # 衡量刚刚树形结构为每个batch生成的action
                    recnum,reward = self.env[j].get_reward_tpgr(sampled_action[j])
                    ars[0][step_count].append(sampled_action[j]) # 往后面添策略网络选出来的用户id
                    ars[1][step_count].append(reward)
                    ars[2][step_count].append(recnum)

                # 停止条件；这里设置的是产生了episode长度的action
                # print("step_count: ",step_count)
                if step_count==self.episode_length:
                    break

        #采样到了一整条episode； 看看结果如何
        # 只需要reward衡量指标哦
        print("=" * 80)
        print('training step: %d' % (self.training_steps))

        reward_list = np.transpose(np.array(ars[2])) #RecNum
        print("evaluate 每个batch选指定人数的RecNum",reward_list[:,-1])
        print("MaxRecNum:", np.amax(reward_list, axis=1), "Avg：", np.mean(np.amax(reward_list, axis=1)))
        # print("每个batch最大RecNum:", np.amax(reward_list, axis=1),"平均值：",np.mean(np.amax(reward_list, axis=1)))
        print("每个batch最大RecNum出现的位置:",np.argmax(reward_list, axis=1))

        # 计算所有batch的平均reward(RecNum)
        # train_ave_reward = np.mean(reward_list[:self.episode_length]) #算平均reward
        # train_ave_reward = np.mean(reward_list[:self.episode_length],axis=1)  # 算平均reward


        # save the model
        params = self.sess.run(self.W_list + self.b_list)
        model = {'W_list': params[:len(self.W_list)], 'b_list': params[len(self.W_list):], 'result_file_path': self.result_file_path}
        utils.pickle_save(model, self.save_model_path + 's%d' % self.training_steps)

        # save the result
        # self.storage.append([np.mean(train_ave_reward)])
        # utils.pickle_save(self.storage, self.result_file_path)

        # print("max recnum",self.env.max_reward)
        # print('\t一个batch选择一个用户，得到的曝光量的平均:' ,np.around(train_ave_reward, decimals=2))
        # print('\t所有batch选择一个用户，得到的曝光量的平均: %2.4f' %(np.mean(train_ave_reward)))


        del ars
        gc.collect()  # 清内存
        self.is_eval = False

