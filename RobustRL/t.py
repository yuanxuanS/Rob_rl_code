import torch
import numpy as np
import copy

print(torch.exp(torch.tensor(100)))
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