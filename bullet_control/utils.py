import collections

import numpy as np


def flatten(obs):
    if not isinstance(obs, collections.OrderedDict):
        raise TypeError("Only flattening of OrderedDict is supported")
    return np.concatenate([o.flatten() for o in obs])
