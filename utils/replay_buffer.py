from typing import Union, List, Dict

from easydict import EasyDict

import numpy as np
import torch

from utils.utils import to_tensor, flatten


class ReplayBuffer(object):
    """ Replay Memory of transition data.
    Args:
        max_size (int): size of replay memory. If size of data inserted into the buffer exceeds, then the max_size will
            increase according to the new full data size.
    """

    def __init__(self, max_size: int, device: torch.device):
        self.max_size = max_size
        self.device = device

        self._build = False
        self._data = {}

        self.reset()

    def get_batches_starting_indexes(self, batch_size):
        if batch_size >= self._valid_count:
            return [np.arange(self._valid_count)]

        indexes = np.arange(0, self._valid_count, 1)
        indexes = np.random.permutation(indexes)

        num_indexes = batch_size
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def sample_batch_by_indices(self, batch_indices) -> EasyDict:
        return EasyDict({key: value[batch_indices] for key, value in self._data.items()})

    def append(self, data: Union[List[Dict[str, Dict[str, List[np.ndarray]]]], Dict[str, Dict[str, List[np.ndarray]]]]):
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
                    if batch_size > self.max_size:
                        print(f"Enlarge replay buffer size from {self.max_size} to {batch_size}")
                        self.max_size = batch_size
                    buffer = torch.zeros((self.max_size, *value.shape[1:]),
                                         dtype=torch.float32).to(self.device)
                    buffer[: batch_size] = value
                    self._data[key] = buffer

            self._build = True
        else:
            valid_key = next(iter(self._data.keys()))
            batch_size = len(data[valid_key])

            new_data_size = self._current_pos + batch_size
            if new_data_size > self.max_size:  # 溢出,自动扩展
                print(f"Enlarge replay buffer size from {self.max_size} to {new_data_size}")
                self.max_size = new_data_size

                new_data = {}
                for key, old_data in self._data.items():
                    new_data[key] = torch.zeros((self.max_size, *old_data.shape[1:]),
                                                dtype=torch.float32).to(self.device)
                    new_data[key][: self._current_pos] = old_data[self._current_pos]
                self._data = new_data

            for key, buffer in self._data.items():
                value = to_tensor(data[key], raise_error=True).to(self.device)
                buffer[self._current_pos: self._current_pos + batch_size] = value

        self._valid_count = min(self._valid_count + batch_size, self.max_size)  # 当前size
        # print(f"Replay Buffer: data position: [{self._current_pos} : {self._valid_count}], size: {batch_size}")
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
