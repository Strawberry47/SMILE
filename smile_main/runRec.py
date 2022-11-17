#coding: utf-8
import sys

from src.StateTracker import StateTrackerGRU

sys.path.append(".")
import tensorflow as tf
from UserSelector import UserSelector
from tree import Tree
import utils
import torch
import numpy as np
import os

# # 定义TensorFlow配置
# config_control = tf.ConfigProto()
# # 配置GPU内存分配方式，按需增长，很关键
# config_control.gpu_options.allow_growth = True
# # 配置可使用的显存比例
# config_control.gpu_options.per_process_gpu_memory_fraction = 0.1


class Recommender():
    def __init__(self, config): # 传入了配置文件
        self.config = config
        # set seed
        seed = 2022
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        GPU = torch.cuda.is_available()
        self.device = torch.device('cuda' if GPU else "cpu")

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    def run(self):
        # 获取配置文件中的值
        max_training_step = int(self.config['META']['MAX_TRAINING_STEP']) # 50
        log_step = int(self.config['META']['LOG_STEP'])
        log = utils.Log() # 用于输出日志

        # constructing tree
        # if self.config['TPGR']['CONSTRUCT_TREE'] == 'T':
        #     log.log('start constructing tree', True)
        #     tree = Tree(self.config)
        #     tree.construct_tree()
        #     log.log('end constructing tree', True)

        # training model
        log.log('start training src', True) # 输出日志文件和当前时间
        src = UserSelector(self.config, self.sess).to(self.device)

        user_embedding = np.loadtxt(self.config['ENV']['OUT_PUT'] + 'user_embedding_dim%d' % int(self.config['META']['ACTION_DIM']), dtype=float, delimiter='\t')
        item_embedding = np.loadtxt(self.config['ENV']['OUT_PUT'] + 'item_embedding_dim%d' % int(self.config['META']['ITEM_DIM']), dtype=float, delimiter='\t')
        state_tracker = StateTrackerGRU(input_dim=int(self.config['META']['ACTION_DIM']), hidden_dim=12,
                                        output_dim=int(self.config['META']['STATE_DIM']),
                                        seed=2022, user_emb=user_embedding,item_emb=item_embedding,
                                        MAX_TURN=int(self.config['META']['EPISODE_LENGTH'])+1, device=self.device).to(self.device)



        for i in range(0, max_training_step): #
            if i % log_step == 0: # 多久运行一次evaluate
                src.evaluate()
                log.log('evaluated\n', True)
            src.train()
        log.log('end training src')
