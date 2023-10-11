import random
import numpy as np

class ER:
    """ Experience Replay """
    def __init__(self, capacity):
        self.capacity = capacity
        self._data = []

    def append(self, sample, error=0):
        """
        sample: (s, a, s', r)
        error: td error/priority value
        """
        self._data.append(sample)

    def sample(self, batch_size=32, global_step=0):
        idxs = np.random.choice(len(self._data), batch_size, replace=False).tolist()
        batch = [self._data[i] for i in idxs]
        is_weight = 0  # there is no importance weight (PER)
        return batch, idxs, is_weight

    def __len__(self):
        return len(self._data)

    def _last_transit(self):
        return self._data[-1]

    def _temp_update_idxs(self, idxs):
        return idxs.append(len(self._data)-1)

