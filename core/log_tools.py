# -*- coding: utf-8 -*-
# @Time    : 2022/6/6 20:02
# @Author  : Shiqi Wang
# @FileName: log_tools.py
import numpy as np
from numbers import Number
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Tuple, Union, Callable, Optional
from tensorboard.backend.event_processing import event_accumulator


WRITE_TYPE = Union[int, Number, np.number, np.ndarray]


class BaseLogger(ABC):
    """The base class for any logger which is compatible with trainer."""

    def __init__(self, writer: Any) -> None:
        super().__init__()
        self.writer = writer

    @abstractmethod
    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
        """Specify how the writer is used to log data.

        :param str key: namespace which the input data tuple belongs to.
        :param int x: stands for the ordinate of the input data tuple.
        :param y: stands for the abscissa of the input data tuple.
        """
        pass

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        pass

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        pass

    def restore_data(self) -> Tuple[int, int, int]:
        """Return the metadata from existing log.

        If it finds nothing or an error occurs during the recover process, it will
        return the default parameters.

        :return: epoch, env_step, gradient_step.
        """
        pass


class BasicLogger(BaseLogger):
    """A loggger that relies on tensorboard SummaryWriter by default to visualize \
    and log statistics.

    You can also rewrite write() func to use your own writer.

    :param SummaryWriter writer: the writer to log data.
    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data(). Default to 1000.
    :param int save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
    ) -> None:
        super().__init__(writer)
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_save_step = -1

    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
        self.writer.add_scalar(key, y, global_step=x)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.

        .. note::

            ``collect_result`` will be modified in-place with "rew" and "len" keys.
        """
        if collect_result["n/ep"] > 0:
            collect_result["rew"] = collect_result["rews"].mean()
            collect_result["recnum"] = collect_result["recnum"].mean()
            collect_result["len"] = collect_result["lens"].mean()

            if step - self.last_log_train_step >= self.train_interval:
                self.write("train/n/ep", step, collect_result["n/ep"])
                self.write("train/rew", step, collect_result["rew"])
                self.write("train/recnum", step, collect_result["recnum"])
                self.write("train/len", step, collect_result["len"])
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.

        .. note::

            ``collect_result`` will be modified in-place with "rew", "rew_std", "len",
            and "len_std" keys.
        """
        assert collect_result["n/ep"] > 0
        rews, lens, recnums = collect_result["rews"], collect_result["lens"], collect_result["recnum"]
        recnum, recnum_std = recnums.mean(),recnums.std()
        rew, rew_std, len_, len_std = rews.mean(), rews.std(), lens.mean(), lens.std()
        collect_result.update(rew_mean=rew, rew_std=rew_std, recnum=recnum,recnum_std=recnum_std,
                              average_rew=float('%.2f' % (rew/len_)),len=len_, len_std=len_std)
        if step - self.last_log_test_step >= self.test_interval:
            self.write("test/recnum", step, recnum)
            self.write("test/recnum_std", step, recnum_std)
            self.write("test/rew_mean", step, rew)
            self.write("test/rew_std", step, rew_std)
            self.write("test/average_rew", step, float('%.2f' % (rew/len_)))
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        if step - self.last_log_update_step >= self.update_interval:
            for k, v in update_result.items():
                self.write(k, step, v)
            self.last_log_update_step = step

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)
            self.write("save/epoch", epoch, epoch)
            self.write("save/env_step", env_step, env_step)
            self.write("save/gradient_step", gradient_step, gradient_step)

    def restore_data(self) -> Tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("save/epoch")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = ea.scalars.Items("save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step


class LazyLogger(BasicLogger):
    """A loggger that does nothing. Used as the placeholder in trainer."""

    def __init__(self) -> None:
        super().__init__(None)  # type: ignore

    def write(self, key: str, x: int, y: WRITE_TYPE, **kwargs: Any) -> None:
        """The LazyLogger writes nothing."""
        pass
