import random

import torch
import numpy as np
import copy
from scipy.stats import beta
import matplotlib.pyplot as plt


# ---beta distri
# a_lst = [1.1, 2, 3, 4, 5]  # 贝塔分布的alpha值
# b_lst = [1.1, 2, 3, 4, 5]  # 贝塔分布的beta值
# x = np.arange(0.01, 1, 0.01)  # 给定的输入数据
# round = 2
# i = round * len(a_lst)
# for a in a_lst:
#     i += 1
#     plt.figure()
#     for b in a_lst:
#         y = beta.pdf(x, a, b)
#         plt.plot(x, y, label=f'a={a} b={b}')
#         plt.legend()
#     plt.savefig("./beta/"+str(i)+".jpg")
# plt.show()

alpha_lst = np.arange(0.01, 5, 0.1)
beta_lst = np.arange(0.01, 5, 0.1)

for alpha in alpha_lst:
    for beta in beta_lst:
        m = torch.distributions.beta.Beta(alpha, beta)
        lst = []
        for i in range(3000):
            lst.append(m.sample())
        print(max(lst), min(lst))
# print(torch.exp(torch.tensor(100)))
# print(torch.rand(1))
# a = torch.Tensor([2, 3])
#
# a2 = torch.Tensor([[2, 3]])
#
# a_np = a.numpy()
# print(f"id a {id(a.data)} np {id(a_np.base)}")
# print(f"before a {a} dtype {a.dim()} {a.size()[0]} a_np {a_np.ndim} size {a_np.size}")
# print(f"{a + a_np}")        # tensor

# print(f"{torch.add(a, a_np).float()}")
# a = [2, 3]
# def func(a):
#     # print(f"dim is {a.dim()}")
#     a_cp = copy.copy(a)
#     a_cp = copy.deepcopy(a)
#     a_cp[0] = 1
#     print(f"in func {a}")
#
# func(a)
# print(f"out func {a}")

# b = torch.Tensor([1, 2, 3])
# c = a * b
# print(a)
# print(b)
# print(c)
# reward_ = [1, 2]
# reward_ = list(np.add(reward_, [8.]*len(reward_)) / 8.)
# print(reward_)
#
# # print(type([1, 2] * np.ones(2) * 2))
# tensor1  = torch.tensor([1, 2])
# tensor2 = torch.tensor([2, 2])
# print(tensor1 + tensor2)
#
# array1 = np.array([1,2])
# array2 = np.array([2, 3])
# print(array1 + array2)
# print(torch.rand(1))