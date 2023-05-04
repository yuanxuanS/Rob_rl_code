from copy import deepcopy

import random
import numpy as np
import networkx as nx
# from src.agent.baseline import *

import pandas as pd
print_tag = "IC---"

def runIC(G, S, p=0.1):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial list of vertices
    p -- propagation probability, float or one-dimension vector
    Output: T -- resulted influenced set of vertices (including S)
    '''
    
    T = deepcopy(S)
    # print(f"{print_tag} T is {T} its type is {type(T)}")
    for u in T:
        # print(f"{print_tag} G  is {G}")
        for v in G[u]:  ## 遍历该节点的所有邻节点， 字典中的key:value, 遍历时v为key这里即邻节点
            w = 1      ## 因为每个seed的邻节点都会被遍历到，所以这里只计算这一条连接就行？？
            if isinstance(p, float):
                prob = 1 - (1-p)**w
            else:  # 网络传播概率不同
                p_one_edge = p[u, v]
                # print(f'this proba is {p_one_edge}')
                prob = 1 - (1-p_one_edge)**w
            if v not in T and random.random() < prob:
                T.append(v)
    return T

def runIC_repeat(G, S, p=0.01, sample=1000):
    infl_list = []

    # with no seeds
    if len(S) == 0:
        print(f"{print_tag} no seed to influence")
        return 0., 0.
    for i in range(sample):
        T = runIC(G, S, p=p)
        influence = len(T)      # 最后的影响力值为激活的node总数，包括seeds中的
        infl_list.append(influence)
    infl_mean = np.mean(infl_list)
    infl_std = np.std(infl_list)

    return infl_mean, infl_std 

