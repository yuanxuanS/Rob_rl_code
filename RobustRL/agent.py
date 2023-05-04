from models import GAT
import torch
import numpy as np
import random
import copy


class DQAgent:
    def __init__(self, env, lr, model_name,
                 init_epsilon, train_batch, update_target_steps):

        self.env = env
        self.graph = self.env.graph  # a graph, Graph_IM instance
        self.z = self.env.z
        self.model_name = model_name
        self.node_features_dims = self.env.node_feat_dimension
        self.node_features = self.env.node_features
        # policy
        self.curr_epsilon = init_epsilon

        if self.model_name == 'GAT_QN':
            # args
            features_dim = self.node_features_dims
            hidden_dim = 4

            alpha = 0.2  # leakyReLU的alpha
            nhead = 1

            self.policy_model = GAT(nfeat=features_dim, nhid=hidden_dim, nout=1,  alpha=alpha,
                                    nheads=nhead, mergeZ=True, mergeState=True)
            self.target_model = GAT(nfeat=features_dim, nhid=hidden_dim, nout=1,  alpha=alpha,
                                    nheads=nhead, mergeZ=True, mergeState=True)

            with torch.no_grad():
                self.target_model.load_state_dict(self.policy_model.state_dict())
        # buffer
        self.memory = []

        # train args
        self.train_batch_size = train_batch     # 训练网络需要的样本数量
        self.episode_steps = self.env.sub_T
        self.gamma = 0.99
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.copy_model_steps = update_target_steps
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)

        # test
        self.print_tag = "DQN Agent---"

    def reset(self):
        self.iter_step = 1
        print(f"{self.print_tag} agent reset done!")

    def act(self, observation, feasible_action):
        # policy

        if self.curr_epsilon > np.random.rand():
            action = np.random.choice(feasible_action)
            # print(f"action is {action}")
        else:
            # GAT, 输入所有节点特征， 图的结构关系-邻接矩阵，
            # node_features 融合state
            # print(f"{self.print_tag} adj_matrix is {self.graph.adj_matrix}")
            input_node_feat = copy.deepcopy(self.node_features)
            q_a = self.policy_model(input_node_feat, self.graph.adj_matrix, observation, z=self.z)
            infeasible_action = [k for k in range(self.graph.node) if k not in feasible_action]
            print(f"{self.print_tag} infeasible action is {infeasible_action}")
            q_a[infeasible_action] = -9e15
            # print(f"{self.print_tag} final q_a is {q_a}")
            action = q_a.argmax()


        if not isinstance(action, int):
            action = int(action)
        # ？ return action.item()
        return action

    def remember(self, sample_lst):
        '''

        :param sample_lst: [state, action, reward, next state]
        :return:
        '''
        self.memory.append(sample_lst)

    def get_sample(self):
        if len(self.memory) > self.train_batch_size:
            batch = random.sample(self.memory, self.train_batch_size)
            print(f"{self.print_tag} batch type is {type(batch)} and batch is {batch}")
            # print(f" zip is {list(zip(*batch))}")
            state_batch = list(list(zip(*batch))[0])
            print(f"state batch is {state_batch}")
            action_batch = list(list(zip(*batch))[1])
            reward_batch = list(list(zip(*batch))[2])
            next_state_batch = list(list(zip(*batch))[3])
        else:
            batch = []
        return batch
        # return state_batch, action_batch, reward_batch, next_state_batch

    def update(self):
        # 采样batch更新policy_model
            # 从memory中采样
        batch = self.get_sample()
        if not batch:
            print(f"{self.print_tag} no enough sample and no update")
            return 0.

        losses = []
        for transition in batch:
            state, action, reward, next_state = transition
            # 用目标网络计算目标值y
            if self.iter_step == self.episode_steps:
                target = reward
            else:
                target = reward + self.gamma * self.target_model(self.node_features, self.graph.adj_matrix, next_state, z=self.z).max()

            if not isinstance(target, torch.Tensor):
                target = torch.Tensor([target])
            # print(f"{self.print_tag} calculated target q is {target}")
            # 用行为网络计算当前值q
            q_a = self.policy_model(self.node_features, self.graph.adj_matrix, state, z=self.z)
            q = q_a[action]
            # print(f"{self.print_tag} calculated action q is {q}")
            # y和q计算loss，更新行为网络
            loss_cur = self.criterion(q, target)

            losses.append(loss_cur)
        loss = torch.mean(torch.tensor(losses, requires_grad=True))
        print(f"{self.print_tag} update losses are {losses} and loss is {loss}")
            # 每 C step，更新目标网络 = 当前的行为网络
        if self.iter_step % self.copy_model_steps == 0:
            with torch.no_grad():
                self.target_model.load_state_dict(self.policy_model.state_dict())        #？？ 是这样用吗


        # 梯度更新
        self.loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iter_step += 1     # 每个step进行一次act和一次update policy network，在update时更新agent当前的step
        return self.loss



