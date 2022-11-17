# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 15:00
# @Author  : Shiqi Wang
# @FileName: utils.py
import time
import numpy as np
from typing import Any, Dict, Union, Callable, Optional

import pandas as pd
from numba import njit
from scipy.sparse import csr_matrix
from torch import optim
import torch

from core.base import BasePolicy
from core.collector import Collector
from core.log_tools import BaseLogger


#return test information
# ['reward_mean'],['accumulated_reward']
def test_episode(
    policy: BasePolicy,
    collector: Collector,
    test_fn: Optional[Callable[[int, Optional[int]], None]],
    epoch: int,
    n_episode: int,
    logger: Optional[BaseLogger] = None,
    global_step: Optional[int] = None,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    result = collector.collect(n_episode=n_episode)
    if reward_metric:
        result["rews"] = reward_metric(result["rews"])
    if logger and global_step is not None:
        logger.log_test_data(result, global_step)
    return result


def gather_info(
    start_time: float,
    train_c: Optional[Collector],
    test_c: Collector,
    best_reward: float,
    best_reward_std: float,
) -> Dict[str, Union[float, str]]:
    """A simple wrapper of gathering information from collectors.

    :return: A dictionary with the following keys:

        * ``train_step`` the total collected step of training collector;
        * ``train_episode`` the total collected episode of training collector;
        * ``train_time/collector`` the time for collecting transitions in the \
            training collector;
        * ``train_time/model`` the time for training models;
        * ``train_speed`` the speed of training (env_step per second);
        * ``test_step`` the total collected step of test collector;
        * ``test_episode`` the total collected episode of test collector;
        * ``test_time`` the time for testing;
        * ``test_speed`` the speed of testing (env_step per second);
        * ``best_reward`` the best reward over the test results;
        * ``duration`` the total elapsed time.
    """
    duration = time.time() - start_time
    model_time = duration - test_c.collect_time
    test_speed = test_c.collect_step / test_c.collect_time
    result: Dict[str, Union[float, str]] = {
        "test_step": test_c.collect_step,
        "test_episode": test_c.collect_episode,
        "test_time": f"{test_c.collect_time:.2f}s",
        "test_speed": f"{test_speed:.2f} step/s",
        "best_reward": best_reward,
        "best_result": f"{best_reward:.2f} ± {best_reward_std:.2f}",
        "duration": f"{duration:.2f}s",
        "train_time/model": f"{model_time:.2f}s",
    }
    if train_c is not None:
        model_time -= train_c.collect_time
        train_speed = train_c.collect_step / (duration - test_c.collect_time)
        result.update({
            "train_step": train_c.collect_step,
            "train_episode": train_c.collect_episode,
            "train_time/collector": f"{train_c.collect_time:.2f}s",
            "train_time/model": f"{model_time:.2f}s",
            "train_speed": f"{train_speed:.2f} step/s",
        })
    return result

def minibatch(*tensors, **kwargs):

    batch_size = 2048

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0])) # 传入的第一个参数是tuple类型
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1: # 只传入了一个参数
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

class BPRLoss:
    def __init__(self,
                 recmodel):
        self.model = recmodel
        self.weight_decay = 0.0001
        self.lr = 0.001
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()


@njit
def find_negative(user_ids, item_ids, mat, df_negative, max_item):
    for i in range(len(user_ids)):
        user, item = user_ids[i], item_ids[i]  # 一条一条地取

        neg = item + 1
        while neg <= max_item:
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

def negative_sample(df):
    sparseMatrix = csr_matrix((np.ones(len(df)), (df['userid'], df['itemid'])),
                              shape=(df['userid'].max() + 1, df['itemid'].max() + 1),
                              dtype=np.bool).toarray()

    df_negative = np.zeros([len(df), 2])
    find_negative(df['userid'].to_numpy(), df['itemid'].to_numpy(), sparseMatrix, df_negative,
                  df['itemid'].max())
    df_negative = pd.DataFrame(df_negative, columns=["userid", "neg_itemid"], dtype=int)
    pairwise = pd.concat([df[['userid'] + ['itemid']], df_negative['neg_itemid']], axis=1)

    users = torch.Tensor(pairwise['userid'].to_numpy()).long()
    posItems = torch.Tensor(pairwise['itemid'].to_numpy()).long()
    negItems = torch.Tensor(pairwise['neg_itemid'].to_numpy()).long()
    return  users,posItems,negItems
