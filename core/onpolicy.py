# -*- coding: utf-8 -*-
# @Time    : 2022/6/4 11:38
# @Author  : Shiqi Wang
# @FileName: onpolicy.py.py

import time
import torch
import tqdm
import warnings
import numpy as np
from collections import defaultdict
from typing import Dict, Union, Callable, Optional
try:
    from tensorflow.python.keras.callbacks import CallbackList
except ImportError:
    from tensorflow.python.keras._impl.keras.callbacks import CallbackList

from core.base import BasePolicy
from core.collector import Collector
from tianshou.utils import tqdm_config, MovAvg
from core.utils import test_episode, gather_info
from core.log_tools import BaseLogger,LazyLogger

def onpolicy_trainer(
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        max_epoch: int,
        step_per_epoch: int,
        repeat_per_collect: int,
        episode_per_test: int,
        batch_size: int,
        step_per_collect: Optional[int] = None,
        episode_per_collect: Optional[int] = None,
        train_fn: Optional[Callable[[int, int], None]] = None,
        test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
        resume_from_log: bool = False,
        reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        logger: BaseLogger = LazyLogger(),
        # callbacks: Optional[list[History]] = None,
        verbose: bool = True,
        test_in_train: bool = True,
        save_model_fn=None,
) -> Dict[str, Union[float, str]]:
    """A wrapper for on-policy trainer procedure.

    The "step" in trainer means an environment step (a.k.a. transition).

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int repeat_per_collect: the number of repeat time for policy learning, for
        example, set it to 2 means the policy needs to learn each given batch data
        twice.
    :param int episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatly in each epoch.
    :param int episode_per_collect: the number of episodes the collector would collect
        before the network update, i.e., trainer will collect "episode_per_collect"
        episodes and do some policy network update repeatly in each epoch.
    :param function train_fn: a hook called at the beginning of training in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy: BasePolicy) ->
        None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to True.

    :return: See :func:`~tianshou.trainer.gather_info`.

    .. note::

        Only either one of step_per_collect and episode_per_collect can be specified.
    """
    if save_fn:
        warnings.warn("Please consider using save_checkpoint_fn instead of save_fn.")

    start_epoch, env_step, gradient_step = 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, gradient_step = logger.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat() # 重置
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    test_result = test_episode(policy, test_collector, test_fn, start_epoch,
                               episode_per_test, logger, env_step, reward_metric)
    best_epoch = start_epoch
    best_average_rew, best_average_rew_std = test_result["average_rew"], test_result["rew_std"]
    best_recnum, best_recnum_std= test_result["recnum"], test_result["recnum_std"]


    # mean_reward, reward_std = test_result["rews"], test_result["rew_std"]
    # print("*"*10,"test result:","*"*10)
    # print("max_cum_reward: {}, mean_cum_reward: {}".format(test_result["rews"].max(),mean_reward))


    # # add callbacks
    callbacks = CallbackList(policy.callbacks)
    callbacks.set_model(policy)
    callbacks.on_train_begin()
    if not hasattr(callbacks, 'model'):  # for tf1.40
        callbacks.__setattr__('model', policy)
    callbacks.model.stop_training = False

    # 训练
    for epoch in range(1 + start_epoch, 1 + max_epoch):
        # train
        policy.train()
        callbacks.on_epoch_begin(epoch)
        torch.autograd.set_detect_anomaly(True)

        with tqdm.tqdm(
                total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config # 重复total次
        ) as t:
            while t.n < t.total:
                if train_fn:
                    train_fn(epoch, env_step)
                # 这里是重点哦！相当于收集轨迹，会输出episode_count, step_count, rews, lens, idxs
                result = train_collector.collect(n_step=step_per_collect,
                                                 n_episode=episode_per_collect)
                if result["n/ep"] > 0 and reward_metric:
                    result["rews"] = reward_metric(result["rews"])
                env_step += int(result["n/st"])
                t.update(result["n/st"]) # 手动更新进度条；更新次数
                logger.log_train_data(result, env_step)
                last_rew = result['rew'] if 'rew' in result else last_rew
                last_len = result['len'] if 'len' in result else last_len
                # the output during training
                data = {
                    "env_step": str(env_step),
                    "rew": f"{last_rew:.2f}",
                    "len": str(int(last_len)),
                    "recnum":str(int(result["recnum"])),
                    "n/ep": str(int(result["n/ep"])),
                    "n/st": str(int(result["n/st"])),
                }
                if result["n/ep"] > 0:
                    if test_in_train and stop_fn and stop_fn(result["rew"]): # 不会运行这里的
                        test_result = test_episode(
                            policy, test_collector, test_fn,
                            epoch, episode_per_test, logger, env_step)
                        if stop_fn(test_result["rew"]):
                            if save_fn:
                                save_fn(policy)
                            logger.save_data(
                                epoch, env_step, gradient_step, save_checkpoint_fn)
                            t.set_postfix(**data)
                            return gather_info(
                                start_time, train_collector, test_collector,
                                test_result["rew"], test_result["rew_std"])
                        else:
                            policy.train()

                # 更新模型哦！
                losses = policy.update(
                    0, train_collector.buffer,
                    batch_size=batch_size, repeat=repeat_per_collect)

                step = max([1] + [
                    len(v) for v in losses.values() if isinstance(v, list)])
                gradient_step += step
                for k in losses.keys():
                    stat[k].add(losses[k])
                    losses[k] = stat[k].get()
                    data[k] = f"{losses[k]:.3f}"
                logger.log_update_data(losses, gradient_step)

                display = {"total_env_step": int(float(data["env_step"])),
                           # "len_episode": float(data["n/st"]) / float(data["n/ep"]),
                           "average_rew": "{:.3f}".format(float(data["rew"]) / float(data["n/ep"])),
                            "cum_rew": data["rew"], "recnum": data["recnum"]}
                t.set_postfix(**display) # 设置进度条右边显示的信息

            if t.n <= t.total:
                t.update()
        # test
        test_result = test_episode(policy, test_collector, test_fn, epoch,
                                   episode_per_test, logger, env_step, reward_metric)
        # mean_cumreward
        avg_rew, rew_std = test_result["average_rew"], test_result["rew_std"]
        recnum, recnum_std = test_result["recnum"], test_result["recnum_std"]

        if best_epoch < 0 or best_recnum < recnum:
            best_epoch, best_average_rew, best_average_rew_std, best_recnum, best_recnum_std\
                = epoch, avg_rew, rew_std, recnum, recnum_std
            if save_fn:
                save_fn(policy)
        logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)

        test_result['best_average_rew'] = best_average_rew
        test_result['best_recnum'] = best_recnum

        callbacks.on_epoch_end(epoch, test_result)

        if verbose:
            print(f"Epoch #{epoch}: test_average_reward: {avg_rew:.6f} ± {rew_std:.6f}, test_recnum"
                  f": {recnum:.6f} ± {recnum_std:.6f},best_average_rew"
                  f": {best_average_rew:.6f} ± {best_average_rew_std:.6f}, best_rencum"
                  f": {best_recnum:.6f} ± {best_recnum_std:.6f} in #{best_epoch}")
        if stop_fn and stop_fn(best_average_rew):
            break

    callbacks.on_train_end()

    return gather_info(start_time, train_collector, test_collector,
                       best_average_rew, best_recnum)
