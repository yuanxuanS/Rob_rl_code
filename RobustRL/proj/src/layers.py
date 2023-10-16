import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha, concat=True, mergeZ=False, node_dim=None, method="base"):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.method = method

        self.W = nn.Parameter(torch.empty(self.in_features, self.out_features))    ## W [in_f, out_f]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))    ## a 权重向量 [2*out_f, 1]
        # print(f"----- a max is {self.a.max()} min is {self.a.min()}")
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.method == "aggre_degree":
            self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
            nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
            self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
            nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)

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
    def forward(self, h, adj, s_mat, z=None):
        # print(f"{self.print_tag} -- foward --- before Wmap, h is {h} W is {self.W}")
        # print(f"{self.print_tag} adj is {adj}")
        # print(f"{self.print_tag} adj size {adj.size()}")
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        torch.set_printoptions(profile="full")
        # logging.debug(f" --  W \n {self.W}")
        # logging.debug(f" --  h \n {h}")
        # logging.debug(f" --  Wh \n {Wh}")
        torch.set_printoptions(profile="default")

        e = self._prepare_attentional_mechanism_input(Wh, h, z)   # 注意力系数, [N, N]
        # logging.debug(f" -- e \n {e}")


        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)       # 相邻则为e系数， 否则为负无穷 [N, N]
        torch.set_printoptions(profile="full")
        # logging.debug(f" -- adj is \n {adj}")
        # logging.debug(f" -- before softmax\n {attention}")
        torch.set_printoptions(profile="default")

        attention = F.softmax(attention, dim=1)       # 要考虑degree
        torch.set_printoptions(profile="full")
        # logging.debug(f" --attenion: after softmax \n {attention}")
        # logging.debug(f" -- after clamp \n {attention}")
        # logging.debug(f" -- after sigmoid \n {attention}")

        torch.set_printoptions(profile="default")
        # 去除-9e15带来的爆炸
        # attention = attention.clamp(-1e15)    # wrong 926.txt
        # attention = torch.sigmoid(attention)
        if self.method == "base":
            torch.set_printoptions(profile="full")
            # logging.debug(f" -- after softmax \n {attention}")
            # logging.debug(f" -- after clamp \n {attention}")
            # logging.debug(f" -- after sigmoid \n {attention}")

            torch.set_printoptions(profile="default")

            h_prime = torch.matmul(attention, Wh)  # 结合邻节点信息后，更新的特征。[N, out_f] 是邻节点才进行加权相加。
        elif self.method == "aggre_degree":
            # s
            # logging.debug(f"aggregate degree, \n {s_mat}")
            s = s_mat
            # logging.debug(f"w_ei: {self.W_ei}, w_si: {self.W_si}")
            # logging.debug(f"before aggre s : \n {e}")

            attention = abs(self.W_ei) * attention + abs(self.W_si) * s
            # logging.debug(f"-- attention: after aggre s : \n {attention}")


        torch.set_printoptions(profile="full")
        # logging.debug(f" curr Wh is \n {attention}")
        # logging.debug(f" curr Wh is \n {Wh}")
        # logging.debug(f" -- final h_prime \n {h_prime}")
        torch.set_printoptions(profile="default")
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
        # logging.debug(f"a: \n {a}")
        Wh1 = torch.matmul(Wh, a[:length, :])
        # logging.debug(f"Wh1: \n {Wh1}")
        Wh2 = torch.matmul(Wh, a[length:, :])
        # logging.debug(f"Wh2: \n {Wh2}")
        # broadcast add
        e = Wh1 + Wh2.T
        return e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayer_struc(nn.Module):
    """
    """
    def __init__(self, in_features, out_features, s_in_feat, s_out_feat, alpha, concat=True, mergeZ=False, node_dim=None, method="base"):
        super(GraphAttentionLayer_struc, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.s_in_feat = s_in_feat
        self.s_out_feat = s_out_feat
        self.alpha = alpha
        self.concat = concat
        self.method = method

        self.W = nn.Parameter(torch.empty(self.in_features, self.out_features))    ## W [in_f, out_f]
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # GAT区别：

        self.W_struc = nn.Parameter(torch.empty(self.s_in_feat, self.s_out_feat))  ## W_struc [node_nbr, node_nbr]
        nn.init.xavier_uniform_(self.W_struc.data, gain=1.414)


        self.a = nn.Parameter(torch.empty(size=(2*self.out_features, 1)))    ## a 权重向量 [2*out_f, 1]
        # print(f"----- a max is {self.a.max()} min is {self.a.min()}")
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if self.method == "aggre_degree":
            self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
            nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
            self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
            nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)

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
    def forward(self, h, adj, h_struc, z=None):
        '''
        s_mat: normalized adjacent node degree [n, n]
        '''
        # print(f"{self.print_tag} -- foward --- before Wmap, h is {h} W is {self.W}")
        # print(f"{self.print_tag} adj is {adj}")
        # print(f"{self.print_tag} adj size {adj.size()}")
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        torch.set_printoptions(profile="full")
        # logging.debug(f" --  W \n {self.W}")
        # logging.debug(f" --  h \n {h}")
        # logging.debug(f" --  Wh \n {Wh}")
        torch.set_printoptions(profile="default")

        e = self._prepare_attentional_mechanism_input(Wh, h, z)   # 注意力系数, [N, N]
        # logging.debug(f" -- e \n {e}")


        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)       # 相邻则为e系数， 否则为负无穷 [N, N]
        torch.set_printoptions(profile="full")
        # logging.debug(f" -- adj is \n {adj}")
        # logging.debug(f" -- before softmax\n {attention}")
        torch.set_printoptions(profile="default")

        attention = F.softmax(attention, dim=1)       # 要考虑degree
        torch.set_printoptions(profile="full")
        # logging.debug(f" --attenion: after softmax \n {attention}")
        # logging.debug(f" -- after clamp \n {attention}")
        # logging.debug(f" -- after sigmoid \n {attention}")

        torch.set_printoptions(profile="default")
        # 去除-9e15带来的爆炸
        # attention = attention.clamp(-1e15)    # wrong 926.txt
        # attention = torch.sigmoid(attention)
        if self.method == "base":
            torch.set_printoptions(profile="full")
            # logging.debug(f" -- after softmax \n {attention}")
            # logging.debug(f" -- after clamp \n {attention}")
            # logging.debug(f" -- after sigmoid \n {attention}")

            torch.set_printoptions(profile="default")

            # 和GAT区别：节点特征不是content， 而是degree feature； attention系数计算方式仍一样
            print(f"s mat is \n{h_struc}")
            Wh_struc = torch.mm(h_struc, self.W_struc)        # [n, in_s] * [in_s, out_s]- [n, out_s]
            print(f"wh struc size is {Wh_struc.size()}")
            h_prime = torch.matmul(attention, Wh_struc)  # [n, n] * [n, out_s] - [n, out_s]
            print(f"final h prime size is {Wh_struc.size()}")

        elif self.method == "aggre_degree":
            # s
            # logging.debug(f"aggregate degree, \n {s_mat}")
            s = s_mat
            # logging.debug(f"w_ei: {self.W_ei}, w_si: {self.W_si}")
            # logging.debug(f"before aggre s : \n {e}")

            attention = abs(self.W_ei) * attention + abs(self.W_si) * s
            # logging.debug(f"-- attention: after aggre s : \n {attention}")


        torch.set_printoptions(profile="full")
        # logging.debug(f" curr Wh is \n {attention}")
        # logging.debug(f" curr Wh is \n {Wh}")
        # logging.debug(f" -- final h_prime \n {h_prime}")
        torch.set_printoptions(profile="default")
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
        # logging.debug(f"a: \n {a}")
        Wh1 = torch.matmul(Wh, a[:length, :])
        # logging.debug(f"Wh1: \n {Wh1}")
        Wh2 = torch.matmul(Wh, a[length:, :])
        # logging.debug(f"Wh2: \n {Wh2}")
        # broadcast add
        e = Wh1 + Wh2.T
        return e

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
