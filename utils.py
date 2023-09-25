import torch
import matplotlib.pyplot as plt
import time

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
