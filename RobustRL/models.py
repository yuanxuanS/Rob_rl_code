import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import networkx as nx
import random
import numpy as np
from layers import GraphAttentionLayer
from generate_node_feature import generate_node_feature
from graph import Graph_IM

class S2V_QN(torch.nn.Module):
    def __init__(self, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T):

        super(S2V_QN, self).__init__()
        self.reg_hidden = reg_hidden
        self.embed_dim = embed_dim
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Linear(1, embed_dim)       # 因为节点的输入仅为其索引，所以输入维度是1？
        print(f"mu_1 is {self.mu_1}")
        #self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        #torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim)
        print(f"mu_2 is {self.mu_2}")
        #self.mu_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        #torch.nn.init.normal_(self.mu_2, mean=0, std=0.01)

        if self.len_pre_pooling > 0:
            self.list_pre_pooling = []
            for i in range(self.len_pre_pooling):
                pre_lin = torch.nn.Linear(embed_dim, embed_dim)
                torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
                self.list_pre_pooling.append(pre_lin)

        if self.len_post_pooling > 0:
            self.list_post_pooling = []
            for i in range(self.len_post_pooling):
                pre_lin = torch.nn.Linear(embed_dim, embed_dim)
                torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
                self.list_post_pooling.append(pre_lin)

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden,bias=True)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1,bias=True)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1,bias=True)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj):

        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]


        for t in range(self.T):
            if t == 0:
                mu_1 = self.mu_1(xv)
                #mu_1 = torch.matmul(xv, self.mu_1)
                # mu_2 = self.mu_2(torch.matmul(adj, mu_init))
                # mu = torch.add(mu_1, mu_2).clamp(0)
                mu = mu_1.clamp(0)


            else:
                mu_1 = self.mu_1(xv)
                #mu_1 = torch.matmul(xv, self.mu_1)

                # before pooling:
                if self.len_pre_pooling > 0:
                    for i in range(self.len_pre_pooling):
                        mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj, mu)

                # after pooling
                if self.len_post_pooling > 0:
                    for i in range(self.len_post_pooling):
                        mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2_ = self.mu_2(mu_pool)
                #mu_2_ = torch.matmul(self.mu_2, mu_pool.transpose(1, 2))
                #mu_2_ = mu_2_.transpose(1, 2)
                mu = torch.add(mu_1, mu_2_).clamp(0)

        # q_1 = self.q_1(torch.sum( mu,dim=1).reshape(minibatch_size,1,self.embed_dim).expand(minibatch_size,nbr_node,self.embed_dim))
        xv = xv.transpose(1, 2)
        q_1 = self.q_1(torch.matmul(xv, mu))
        q_1_ = q_1.clone()
        q_1_ = q_1_.expand(minibatch_size, nbr_node, self.embed_dim)
        ####
        # mat = xv.reshape(minibatch_size, nbr_node).type(torch.ByteTensor)
        # mat = torch.ones(minibatch_size, nbr_node).type(torch.ByteTensor) - mat
        # res = torch.zeros(minibatch_size, nbr_node, nbr_node)
        # res.as_strided(mat.size(), [res.stride(0), res.size(2) + 1]).copy_(mat)
        # mu_ = mu.transpose(1, 2)
        # mu_y = torch.matmul(mu_, res)
        # mu_y = mu_y.transpose(1, 2)
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1_, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_=q_.clamp(0)
            q = self.q(q_)
        return q


# hidden_reg = 0
# embed_dim = 5
# len_pre_pooling = 0
# len_post_pooling = 0
# T = 3
# s2v = S2V_QN(reg_hidden=hidden_reg, embed_dim=embed_dim,
#              len_pre_pooling=len_pre_pooling, len_post_pooling=len_post_pooling, T=T)

class W2V_QN(torch.nn.Module):
    def __init__(self, G, len_pre_pooling, len_post_pooling, embed_dim, window_size, num_paths, path_length, T):

        super(W2V_QN, self).__init__()
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Linear(1, embed_dim)
        print(f"mu 1 is {self.mu_1}")
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim)
        print(f"mu 2 is {self.mu_2}")
        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            self.list_pre_pooling.append(torch.nn.Linear(embed_dim, embed_dim))

        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            self.list_post_pooling.append(torch.nn.Linear(embed_dim, embed_dim))

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim)
        self.q = torch.nn.Linear(2 * embed_dim, 1)

        walks = self.build_deepwalk_corpus(G, num_paths=num_paths,
                                           path_length=path_length, alpha=0)

        # self.model = Word2Vec(walks, vector_size=embed_dim, window=window_size
        #                       , min_count=0, sg=1, hs=1, iter=1, negative=0, compute_loss=True)
        self.model = Word2Vec(walks, vector_size=embed_dim, window=window_size      # 通过walks这些样本来训练得到model
                               , min_count=0, sg=1, hs=1, negative=0, compute_loss=True)

        print(self.model)
    def random_walk(self, G, path_length, alpha=0, rand=random.Random(), start=None):
        ## 随机游走，返回一条路径

        ## 指定开始节点
        if start:
            path = [start]
        else:   # 0节点开始，则任意选一个别的非0节点开始
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.nodes()))]   # 返回一个值

        ## 从邻节点中选择节点
        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0: # 直到节点没有邻节点
                if rand.random() >= alpha:
                    path.append(rand.choice(list(nx.neighbors(G, cur))))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]

    def build_deepwalk_corpus(self, G, num_paths, path_length, alpha=0):
        # 返回多条随机游走的路径
        print(f"in deep walk corpus")
        walks = []

        nodes = list(G.nodes())
        print(f"nodes are {nodes}")
        for cnt in range(num_paths):
            print(f"in get path loop ---- {cnt}")
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk(G, path_length, alpha=alpha, start=node))
                print(f"\tcurrent walks is {walks}")
        return walks

    def forward(self, xv, adj, mu_init):
        print(f"model vacab is {list(self.model.wv.key_to_index)}")    # wv word vector; key_to_index 词汇
        print(f"after map {sorted(list(map(int, list(self.model.wv.key_to_index))))} ")
        print(f" {list(map(str, sorted(list(map(int, list(self.model.wv.key_to_index))))))} ")
        print(f"word vector by word2vec model: \n{self.model.wv[list(map(str, sorted(list(map(int, list(self.model.wv.key_to_index))))))]}")
        mu_w2v = torch.from_numpy(  # 把数组转化成tensor
            np.expand_dims(self.model.wv[list(map(str, sorted(list(map(int, list(self.model.wv.key_to_index))))))], axis=0))
        # 对所有词进行排序，获取对应的向量
        # print(f"mu_w2v is {mu_w2v}")        # 训练后的model的每个词汇，对应的向量
        for t in range(self.T):
            if t == 0:
                mu_1 = self.mu_1(xv)        # 更新mu_1. 输入节点个数，[n, 1] * [1, dim] =经过映射，每个节点变为dim长度的向量, n * dim
                print(f" xv * mu_1 = {mu_1}")
                mu_2 = self.mu_2(torch.matmul(adj, mu_w2v))     #更新mu_2 . adj * mu_w2v =对邻接点的聚合； 再通过线性变换, n * dim
                mu = torch.add(mu_1, mu_2).clamp(0)     # 更新mu . n * dim

            else:
                mu_1 = self.mu_1(xv)

                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(adj, mu)

                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)     # 最后的节点的特征向量

        q_1 = self.q_1(torch.matmul(adj, mu))   # [n, dim] * [dim, dim] = [n, dim]. 邻节点的聚合后的特征向量
        q_2 = self.q_2(mu)  # [n, dim] * [dim, dim] = [n, dim]。mu好像是
        q_ = torch.cat((q_1, q_2), dim=-1)  # [1, n, embed_dim * 2]
        print(f"q_ size {q_.size()}")
        q = self.q(q_)      # 映射每个样本特征为1， 拼接后得到最后的特征向量 [1, n, 1]
        print(f" result after model forward ，size {q.size()}")
        return q

# n = 6
# g = nx.erdos_renyi_graph(n=n, p=0.5)
# len_pre = 0
# len_post = 0
# embed_dim = 4
# window_size = 2
# num_paths = 1
# path_length = 3
# T = 2
#
#
# w2v = W2V_QN(g, len_pre, len_post, embed_dim, window_size, num_paths, path_length, T)
# x = g.nodes()
# x = torch.Tensor(x)
# x = x.reshape(-1, 1)        # [n, 1]
# print(f"nodes are {x}")
# adj = nx.adjacency_matrix(g).todense()
# adj = torch.from_numpy(adj).to(torch.float32)       # [n, n]
# print(f"adj is {adj}")
# mu_init = None
# w2v(x, adj, mu_init)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, alpha, nheads, mergeZ, mergeState):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.nfeat = nfeat
        self.nhid = nhid

        self.nout = nout
        self.alpha = alpha
        self.mergeZ = mergeZ
        self.mergeState = mergeState
        # k多头，有k个头就有k个层。
        # nhid 就是输出的特征大小。 因为GAT的隐藏层hid会输出特征
        # print(f"nhid is {nhid}")
        # print(f"stack multi GAT layer")
        self.attentions = [GraphAttentionLayer(self.nfeat, self.nhid, alpha=self.alpha, concat=True, mergeZ=self.mergeZ, node_dim=self.nfeat) for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            # print(f"第{i}个layer, {str(attention)}")
            self.add_module('attention_{}'.format(i), attention)

        # 会把多头的结果拼接起来，所以是nhid * nheads， 输出n个类
        #################
        # print(f"add final attention layer")
        self.out_att = GraphAttentionLayer(self.nhid * nheads, self.nout, alpha=self.alpha, concat=False, mergeZ=self.mergeZ, node_dim=self.nfeat)

        # seed set feature theta
        self.theta = nn.Parameter(torch.empty(1, self.nfeat))  # 大小和每个节点的feature向量一样
        nn.init.uniform_(self.theta, a=0, b=1)  # 均匀分布

        # test
        self.print_tag = "Models ---"

    def forward(self, x, adj, observation, z=None):

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if not isinstance(adj, torch.Tensor):
            adj = torch.Tensor(adj)
        #融合seed set信息
        if self.mergeState:
            seed_set = [idx for idx in range(len(observation[0])) if observation[0][idx] == 1]
            sdset_mask = torch.zeros([x.size()[0], x.size()[1]])
            sdset_mask[seed_set] += 1.
            sdset_mask = sdset_mask * self.theta
            x = x + sdset_mask
            # print(f"seed set mask is {sdset_mask}")
            # print(f"x after seed set mask is {x}")


        ## x, features [n, feature_size]


        x = torch.cat([att(x, adj, z) for att in self.attentions], dim=1)

        x = F.elu(self.out_att(x, adj, z))


        result = F.log_softmax(x, dim=0)    # 不是分类问题，应该是纵向softmax
        return result


# features_dim = 10
# hidden_dim = 4
# dropout = 0.
# alpha = 0.2     # leakyReLU的alpha
# nhead = 1
# model = GAT(nfeat=features_dim, nhid=hidden_dim, nout=1, dropout=dropout, alpha=alpha, nheads=nhead)
# # test
# graph = Graph_IM(nodes=10, edges_p=0.5)
# adj_matrix = graph.adj_matrix
# # print(f"graph adj matrix {graph.adj_matrix}")
# adj_matrix = torch.Tensor(adj_matrix)
# xv = generate_node_feature(graph, features_dim)
# print(f"node feature vector size {xv}")     # [0-100]
# xv = torch.Tensor(xv)       # torch的输入必须是tensor
# y = model(xv, adj_matrix)
#
# print(f"get y from GAT is {y}")     # [n, 1]
# action = y.argmax()     # node index, int
# print(f"max node idx is {action}")