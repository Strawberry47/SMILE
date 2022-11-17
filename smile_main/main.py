# coding: utf-8
# nohup python -u smile_main.py > tpgr0808.log 2>&1 &
# ps -ef|grep python
import argparse
import datetime
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import gym
from keras.callbacks import History
from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from core.log import create_dir, LoggerCallback_RL
from ProcessingData.dataProcess import DataProcess
from core.collector import Collector
from smile_main.StateTracker import StateTrackerGRU
from core.onpolicy import onpolicy_trainer
from core.pg import PGPolicy
from gym.envs.registration import register
# sys.path.append(".")
from UserSelector import UserSelector
from Tree import Tree
import torch
import numpy as np
import configparser
from torch.utils.tensorboard import SummaryWriter
# from tianshou.utils import BasicLogger
from core.log_tools import BaseLogger,LazyLogger,BasicLogger
import time
import logzero
from logzero import logger
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

def main():
    # 解析配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="config_movielens1M")
    args = parser.parse_known_args()[0]

    path = os.path.join("../config", args.dataset)
    config = configparser.ConfigParser()
    _x = open(path)

    # _x = open('../config/config_movielens100k')
    # _x = open('../config/config_movielens1M')
    # _x = open('../config/config_Ciao')
    config.read_file(_x)
    # output logs
    nowtime = datetime.datetime.fromtimestamp(time.time()).strftime("%Y_%m_%d-%H_%M_%S")
    logger_path = os.path.join(config['ENV']['OUT_PUT'], "logs", "[SMILE]_{}.log".format(nowtime))
    logzero.logfile(logger_path)

    print('dataset path: %s' % config['ENV']['ORIGN_RATING_FILE'])
    print('time: %s' % time.asctime(time.localtime(time.time())))

    # set seed
    seed = 2022
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    GPU = torch.cuda.is_available()
    device = torch.device('cuda:1' if GPU else "cpu")
    # device = 'cpu'
    # processing the dataset
    dataset = DataProcess(config)
    print("item_num:{},user_num:{},ratings:{}".format(dataset.item_num,dataset.user_num,dataset.rating_num))

    # constructing tree
    tree = Tree(config,dataset)
    tree.construct_tree()
    print('end constructing tree')

    # create envs
    train_num = int(config['META']['TRAINNUM'])
    test_num = int(config['META']['TESTNUM'])
    # train_num = test_num = 2 # TODO:test


    register(
        id= 'SmileEnv-v0',  # 'KuaishouEnv-v0',
        entry_point='smile_main.SmileEnv:SmileEnv', # 文件夹名.文件名:类名
        kwargs={"config": config,
                "dataset": dataset,
                "device":device}
    )
    smile_env = gym.make('SmileEnv-v0')

    train_envs = DummyVectorEnv(
        [lambda: gym.make("SmileEnv-v0", ) for _ in range(train_num)])
    test_envs = DummyVectorEnv(
        [lambda: gym.make('SmileEnv-v0') for _ in range(test_num)])
    print("end constructing environments")
    train_envs.seed(seed)
    test_envs.seed(seed)

    # state_tracker
    user_embedding = np.loadtxt(config['ENV']['OUT_PUT'] + 'user_embedding_dim%d' % int(config['META']['ACTION_DIM']), dtype=float, delimiter='\t')
    item_embedding = np.loadtxt(config['ENV']['OUT_PUT'] + 'item_embedding_dim%d' % int(config['META']['ITEM_DIM']), dtype=float, delimiter='\t')
    # state 维度20；GRU输入32，隐藏层12
    state_tracker = StateTrackerGRU(config,input_dim=int(config['META']['ACTION_DIM']), hidden_dim=12,
                                    output_dim=int(config['META']['STATE_DIM']),
                                    seed=seed, user_emb=user_embedding,item_emb=item_embedding,
                                    EPISODE_LENGTH=int(config['META']['EPISODE_LENGTH'])+1, device=device).to(device)

    # user selector
    user_selector = UserSelector(config,tree.child_num,device).to(device)

    # orthogonal initialization
    for m in list(user_selector.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    # 指定优化器
    optim_RL = torch.optim.Adam(
        list(user_selector.parameters()), lr=float(config['META']['LEARNING_RATE']))
    optim_state = torch.optim.Adam(state_tracker.parameters(), lr=float(config['META']['LEARNING_RATE']))
    optim = [optim_RL, optim_state]


    policy = PGPolicy(user_selector, optim, dist_fn=torch.distributions.Categorical,
                      discount_factor=float(config['META']['DISCOUNT_FACTOR']),
                      reward_normalization=1,
                      action_space=smile_env.action_space)

    # collector
    # 5000代表max_size
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(5000, len(train_envs)),
        preprocess_fn=state_tracker.build_state)

    test_collector = Collector(policy, test_envs,
        preprocess_fn=state_tracker.build_state)

    log_path = os.path.join(config['ENV']['OUT_PUT'], 'pg')
    writer = SummaryWriter(log_path) # 将数据以特定的格式存储到刚刚提到的那个文件夹中
    logger1 = BasicLogger(writer, save_interval=1000)
    policy.callbacks = [History()] + [LoggerCallback_RL(logger_path)]


    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, max_epoch=100,
        episode_per_collect=train_num,episode_per_test=test_num,
        step_per_epoch=5000, repeat_per_collect = 2, batch_size=1024,logger=logger1)

def save_model_fn(epoch, policy, model_save_path, optim, state_tracker, is_save=False):
    if not is_save:
        return
    model_save_path = model_save_path[:-3] + "-e{}".format(epoch) + model_save_path[-3:]
    # torch.save(model.state_dict(), model_save_path)
    torch.save({
        'policy': policy.state_dict(),
        'optim_RL': optim[0].state_dict(),
        'optim_state': optim[1].state_dict(),
        'state_tracker': state_tracker.state_dict(),
    }, model_save_path)

if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    print("total run time: %f min" %((e-s)/60))
