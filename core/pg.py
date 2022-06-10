# -*- coding: utf-8 -*-
# @Time    : 2022/6/4 11:33
# @Author  : Shiqi Wang
# @FileName: pg.py.py
import torch
import numpy as np
from typing import Any, Dict, List, Type, Union, Optional
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd
from tianshou.data import Batch, ReplayBuffer, to_torch_as


class PGPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        # optim: torch.optim.Optimizer,
        optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        deterministic_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(action_scaling=action_scaling,
                         action_bound_method=action_bound_method, **kwargs)
        self.actor = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        v_s_ = np.full(indice.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indice, v_s_=v_s_, gamma=self._gamma, gae_lambda=1.0)
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, actions = self.actor(batch.obs,self.training)
        act = actions
        dist = self.dist_fn(logits)
        # act = dist.sample()
        return Batch(logits=logits, act=act, state=None, dist=dist)

    # onpolicy中调用哦
    def learn(  # type: ignore
        self, batch, batch_size=0, repeat=0, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        optim_RL, optim_state = self.optim
        for _ in range(repeat):
            optim_state.zero_grad()
            # for b in batch.split(batch_size, merge_last=True):
            batches = batch.split(batch_size, merge_last=True)
            for b_ind, b in enumerate(batches): # 这里面只更新了RL

                # todo:debug!
                # 输入当前state和当前action，返回概率

                log_pro = self.actor.get_pro_by_action(states=b.obs,actions=b.act)
                # result = self(b)
                # dist = result.dist
                # b.act的type放到result.act的device上
                # a = to_torch_as(b.act, result.act)
                # ret是算出来的累积收益哦
                ret = to_torch_as(b.returns, log_pro)
                log_prob = log_pro.reshape(len(ret), -1).transpose(0, 1)
                loss = -(log_prob * ret).mean()
                # 先更新RL
                optim_RL.zero_grad()
                loss.backward(retain_graph=True)
                optim_RL.step()

                losses.append(loss.item())
        optim_state.step()

        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {"loss": losses}
