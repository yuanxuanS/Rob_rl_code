import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha, concat=True, mergeZ=False, node_dim=None):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(self.in_features, self.out_features))    ## W [in_f, out_f]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))    ## a 权重向量 [2*out_f, 1]
        # print(f"----- a max is {self.a.max()} min is {self.a.min()}")
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # 融合z时的参数
        self.mergeZ = mergeZ
        self.node_dim = node_dim
        if self.mergeZ:
            self.V = nn.Parameter(torch.empty(self.in_features, self.node_dim))  ## V [in_f, out_f]
            nn.init.xavier_uniform_(self.V.data, gain=1.414)
            self.z_coeffi = nn.Parameter(torch.rand(1))

        # test
        self.print_tag = "Layers ---"
    def forward(self, h, adj, z=None):
        # print(f"{self.print_tag} -- foward --- before Wmap, h is {h} W is {self.W}")
        # print(f"{self.print_tag} adj is {adj}")
        # print(f"{self.print_tag} adj size {adj.size()}")
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print(f"{self.print_tag} -- foward --- Wh {Wh}")
        e = self._prepare_attentional_mechanism_input(Wh, h, z)   # 注意力系数, [N, N]
        # print(f"{self.print_tag} -- foward --- e {e}")

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)       # 相邻则为e系数， 否则为负无穷 [N, N]
        attention = F.softmax(attention, dim=1)
        # print(f"{self.print_tag} -- foward --attention after softmax \n{attention}")

        h_prime = torch.matmul(attention, Wh)       # 结合邻节点信息后，更新的特征。[N, out_f] 是邻节点才进行加权相加。
        # print(f"{self.print_tag} -- foward --- final h_prime \n {h_prime}")
        if self.concat:
            result = F.elu(h_prime)
            # print(f"{self.print_tag} after elu")
            return result
        else:
            return h_prime  #  [N, out_f]

    def _prepare_attentional_mechanism_input(self, Wh, h=None, z=None):
        # 如果融合z，输入需要的元素h, z
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        e = self._prepare_attentional_like_calc(Wh, self.a)
        # print(f"{self.print_tag} -- foward -- before add e_z {e}")
        if self.mergeZ:
            Vh = torch.mm(h, self.V)
            e_z = self._prepare_attentional_like_calc(Vh, z)
            e += self.z_coeffi * e_z
        # print(f"{self.print_tag} -- foward -- before leakyrelu e {e}")
        return self.leakyrelu(e)

    def _prepare_attentional_like_calc(self, Wh, a):
        # 先类型转换
        if not isinstance(a, torch.FloatTensor):
            if isinstance(a, np.ndarray):
                a = torch.from_numpy(a).float()
        if a.size() != self.a.size():
            a = a.view(-1, 1)
        # print(f"{self.print_tag} {a.size()}")
        length = int(a.size()[0] / 2)
        Wh1 = torch.matmul(Wh, a[:length, :])
        Wh2 = torch.matmul(Wh, a[length:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
