#coding: utf-8
import gym
from gym import spaces

import numpy as np

from smile_main.RewardCalculator import RewardCalculator
# from smile_main.BPR import BPRData
# from algorithm.BPR import BPR
import random
class SmileEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, config, dataset,device='cpu'):
        self.config = config
        self.dataset = dataset
        self.device = device
        self.episode_len = int(self.config['META']['EPISODE_LENGTH'])
        self.dataProcess = dataset
        self.max_turn = int(self.config['META']['EPISODE_LENGTH'])
        self.action_space = spaces.Box(low=0, high=dataset.user_num, shape=(1,), dtype=np.int32)  # user数量
        self.observation_space = spaces.Box(low=0, high=dataset.item_num, shape=(1,), dtype=np.int32)  # item数量
        self.reset()


    # 传入action，输出reward（小矩阵的值）, state等信息
    def step(self, action):
        # action是user列表哦
        self.action=action
        t = self.total_turn # reset中设置为0了哦，这里的t是次数
        done = False
        if t >= (self.max_turn-1):
            done = True

        pos_user_list = self.reward_calculator.userfilter([self.action])
        new_interactions = self.reward_calculator.insertData(self.last_interactions,pos_user_list)
        self.topnlist = self.reward_calculator.train_test(new_interactions,self.device)
        self.last_interactions = new_interactions

        now_reward = self.reward_calculator.getItemCnt(self.topnlist)
        reward = now_reward - self.last_reward
        self.max_recnum = max(self.max_recnum, now_reward)
        self.last_reward = now_reward
        self.total_turn += 1
        self.cum_reward += reward

        # if done:
        #     print("environment done")
        #     print("the last RecNum is: {}, the max RecNum is {}th: {}".format(now_reward,index, self.max_recnum))
            # self.pro_item = self.__item_generator()

        return self.state, reward, done, {'cur_recnum':self.last_reward,
                                          'max_recnum': self.max_recnum}

    def reset(self):
        self.promoted_items = self.__item_generator()
        self.reward_calculator = RewardCalculator(self.config, self.dataProcess,
                                             self.promoted_items,self.device)
        self.last_reward = self.reward_calculator.last_reward
        self.action = None
        self.last_interactions = self.dataProcess.rating
        self.max_recnum = 0
        self.total_turn = 0
        self.max_recnum = 0
        self.cum_reward = 0
        return self.state  # 当前items或者action


    def __item_generator(self):
        item = random.sample(list(range(self.dataset.item_num)),
                                            int(self.dataset.item_num * 0.01))
        return item

    # def __initial_reward_generator(self,reward_calculator):
    #     topnlist = reward_calculator.train_test(self.dataset.rating)
    #     cnt = reward_calculator.getItemCnt(topnlist)
    #     return cnt

    @property
    def state(self):
        if self.action is None:
            res = self.promoted_items
        else:
            res = self.action
        return np.array([res])
