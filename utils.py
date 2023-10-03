import torch
import matplotlib.pyplot as plt
import time
import sys
import psutil
import logging
def test_memory():
    # logging.debug(f"use memory: {psutil.memory_percent()}")
    logging.debug(f"use vitual memory: {psutil.virtual_memory()}")



def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

# t = torch.tensor([[3.2836]])
# compute_advantage(0.9, 0.9, t)

def draw_distri_hist(data_lst, sav_path, name):
    print(f"--- return histogram --- {name}")
    print(f"max is {max(data_lst)}, min is {min(data_lst)}")
    plt.hist(data_lst)
    plt.title(f"histogram of {name}")
    plt.savefig(sav_path + "_" + name)
    plt.close()


def get_size(obj, seen=None):
    """递归计算对象及其嵌套对象的内存大小"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_size(item, seen) for item in obj)
    elif isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__slots__'):
        size += sum(get_size(getattr(obj, slot), seen) for slot in obj.__slots__)

    return size