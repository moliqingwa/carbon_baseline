from typing import Any, Union, AnyStr, List, Dict
import copy

from easydict import EasyDict

import numpy as np
import torch

from utils.utils import to_tensor, flatten


class ReplayBuffer(object):
    """ Replay Memory for saving data.
    Args:
        max_size (int): size of replay memory
    """

    def __init__(self, cfg: EasyDict):
        self._cfg = cfg
        self.max_size = int(self._cfg.size)
        self._deepcopy = self._cfg.deepcopy

        self._build = False
        self._data = {}

        self.reset()

    def sample_batch(self, batch_size: int) -> EasyDict:
        return self.sample_batch_by_indices(self.make_indices(batch_size))

    def make_indices(self, batch_size: int) -> np.ndarray:
        """
        Get sample indices array.

        :param batch_size: The number of the data that will be sampled
        :param sample_range: Buffer slice for sampling, such as `slice(-128, None)`,
         which means only sample among the last 128 data
        :return: An array including all the sample indices
        """
        if batch_size > self._valid_count:
            batch_indices = np.arange(self._valid_count)
        else:
            batch_indices = np.random.randint(self._valid_count, size=batch_size)

        return batch_indices

    def sample_batch_by_indices(self, batch_indices) -> EasyDict:
        return EasyDict({key: copy.deepcopy(value[batch_indices]) if self._deepcopy else value[batch_indices]
                         for key, value in self._data.items()})

    def append(self, data: Union[List[Dict[AnyStr, Dict[AnyStr, List[np.ndarray]]]], Dict[AnyStr, Dict[AnyStr, List[np.ndarray]]]]):
        if isinstance(data, (list, tuple)):   # list of dict to dict of list
            data = [agent_value for env_value in data for agent_value in env_value.values()]
        else:
            data = list(data.values())
        data = {k: np.concatenate(flatten([d[k] for d in data])).astype(np.float32) for k in data[0]}

        batch_size = None
        if not self._build:  # 初始化
            for key, value in data.items():
                value = to_tensor(value, raise_error=False)
                if value is not None:
                    batch_size = len(value) if batch_size is None else batch_size
                    buffer = torch.zeros((self.max_size, *value.shape[1:]),
                                         dtype=torch.float32)
                    buffer[: batch_size] = value
                    self._data[key] = buffer

            self._build = True
        else:
            valid_key = next(iter(self._data.keys()))
            batch_size = len(data[valid_key])

            exceed_size = self._current_pos + batch_size - self.max_size
            if exceed_size > 0:
                self._data = {key: torch.roll(buffer, -exceed_size)
                              for key, buffer in self._data.items()}
                self._current_pos -= exceed_size

            for key, buffer in self._data.items():
                value = to_tensor(data[key], raise_error=True)
                buffer[self._current_pos: self._current_pos + batch_size] = value

        self._valid_count = min(self._valid_count + batch_size, self.max_size)  # 当前size
        print(f"Replay Buffer: data position: [{self._current_pos} : {self._valid_count}], size: {batch_size}")
        self._current_pos = (self._current_pos + batch_size) % self.max_size  # 当前最新的位置

    def count(self) -> int:
        """
        Count how many valid data stored in the buffer.
        :return: Number of valid data.
        """
        return self._valid_count

    def __len__(self):
        return self._valid_count

    def reset(self):
        if self._build:
            pass  # Do nothing, just adjust the position and size.

        self._valid_count = 0
        self._current_pos = 0
