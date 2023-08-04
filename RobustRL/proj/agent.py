from models import GAT
import torch
import numpy as np
import random
import copy
import seed


seed = seed.get_value()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class DQAgent:
    def __init__(self, graph_pool, node_feat_pool, hyper_pool, model_name,
                 main_setting,
                 node_dim,
                 init_epsilon, train_batch, update_target_steps, use_cuda, device):

        self.use_cuda = use_cuda
        self.merge_z = main_setting["observe_z"]
        self.device = device
        self.graphs = graph_pool
        self.node_feat_pool = node_feat_pool
        self.hyper_pool = hyper_pool

        self.graph = None
        self.adj = None
        self.z = None
        self.model_name = model_name

        self.node_features = None
        self.node_features_dims = node_dim
        # policy
        self.curr_epsilon = init_epsilon
        self.policy_model = None
        self.target_model = None

        # buffer
        self.memory = []
        self.buffer_max = 500       # equal to global iterations

        # train args
        self.basic_batch_size = train_batch
        self.train_batch_size = train_batch     # 训练网络需要的样本数量

        self.gamma = main_setting["gamma"]
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.copy_model_steps = update_target_steps
        self.lr = main_setting["lr"]

        if self.model_name == 'GAT_QN':
            hidden_dim = main_setting["hidden_dims"]

            alpha = main_setting["alpha"]  # leakyReLU的alpha
            nhead = main_setting["nheads"]

            self.policy_model = GAT(nfeat=self.node_features_dims, nhid=hidden_dim, nout=1, alpha=alpha,
                                    nheads=nhead, mergeZ=self.merge_z, mergeState=True, use_cuda=self.use_cuda, device=self.device)
            self.target_model = GAT(nfeat=self.node_features_dims, nhid=hidden_dim, nout=1, alpha=alpha,
                                    nheads=nhead, mergeZ=self.merge_z, mergeState=True, use_cuda=self.use_cuda, device=self.device)
            if self.use_cuda:
                self.policy_model.to(self.device)
                self.target_model.to(self.device)
            with torch.no_grad():
                self.target_model.load_state_dict(self.policy_model.state_dict())

            self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
            
        elif self.model_name == "random":
            self.policy_model = None
            self.target_model = None



        # test
        self.print_tag = "DQN Agent---"


    def init_graph(self, g_id):
        self.graph = self.graphs[g_id]  # a graph, Graph_IM instance
        self.adj = torch.Tensor(self.graph.adj_matrix)

    def init_n_feat(self, ft_id):

        self.node_features = torch.Tensor(self.node_feat_pool[ft_id])




    def init_hyper(self, hyper_id):
        self.z = torch.Tensor(self.hyper_pool[hyper_id])

    def reset(self):
        self.iter_step = 1
        # print(f"{self.print_tag} agent reset done!")

    def act(self, observation, feasible_action, mode):
        # policy
        if self.model_name == 'GAT_QN':

            if self.curr_epsilon > np.random.rand() and mode != "valid":
                action = np.random.choice(feasible_action)
                # print(f"action is {action}")
            else:
                # GAT, 输入所有节点特征， 图的结构关系-邻接矩阵，
                # node_features 融合state
                # print(f"{self.print_tag} adj_matrix is {self.graph.adj_matrix}")
                input_node_feat = copy.deepcopy(self.node_features)
                q_a = self.policy_model(input_node_feat.to(self.device), self.adj.to(self.device),
                                        torch.Tensor(observation).to(self.device), z=self.z.to(self.device))
                if self.use_cuda:
                    q_a = q_a.cpu()
                infeasible_action = [k for k in range(self.graph.node) if k not in feasible_action]
                # print(f"{self.print_tag} infeasible action is {infeasible_action}")

                q_a[infeasible_action] = -9e15
                # print(f"{self.print_tag} final q_a is {q_a}")
                action = q_a.argmax()

        elif self.model_name == "random":
            action = np.random.choice(feasible_action)

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
        self.train_batch_size = int((1 + len(self.memory) / self.buffer_max) * self.basic_batch_size)
        if len(self.memory) > self.train_batch_size:

            batch = random.sample(self.memory, self.train_batch_size)
            # print(f"{self.print_tag} batch is {batch}")
            # print(f" zip is {list(zip(*batch))}")
            state_batch = list(list(zip(*batch))[0])
            # print(f"state batch is {state_batch}")
            action_batch = list(list(zip(*batch))[1])
            reward_batch = list(list(zip(*batch))[2])
            next_state_batch = list(list(zip(*batch))[3])
        else:
            batch = []
        return batch
        # return state_batch, action_batch, reward_batch, next_state_batch

    def update(self, i):
        # 采样batch更新policy_model
            # 从memory中采样
        batch = self.get_sample()
        if not batch:
            # print(f"{self.print_tag} no enough sample and no update")
            return 0.

        losses = []
        for transition in batch:
            state, action, reward, next_state, done, g_id, ft_id, hyper_id = transition
            # 用目标网络计算目标值y
            graph = self.graphs[g_id]
            adj = torch.Tensor(graph.adj_matrix)
            node_feature = torch.Tensor(self.node_feat_pool[ft_id])
            hyper = torch.Tensor(self.hyper_pool[hyper_id])
            target = reward + (1 - done) * self.gamma * self.target_model(node_feature.to(self.device), adj.to(self.device),
                                                                          torch.Tensor(next_state).to(self.device), z=hyper.to(self.device)).max()

            if not isinstance(target, torch.Tensor):
                target = torch.Tensor([target])
            # print(f"{self.print_tag} calculated target q is {target}")
            # 用行为网络计算当前值q
            q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                    torch.Tensor(state).to(self.device), z=hyper.to(self.device))
            
            
            q = q_a[action]
            # print(f"{self.print_tag} calculated action q is {q}")
            # y和q计算loss，更新行为网络
            # print(f"q type {q.device}| target {target.device} ")
            loss_cur = self.criterion(q, target)

            losses.append(loss_cur)
        loss = torch.mean(torch.tensor(losses, requires_grad=True))
        # print(f"{self.print_tag} update losses are {losses} and loss is {loss}")

        # 每 C step，更新目标网络 = 当前的行为网络
        if i % self.copy_model_steps == 0:
            with torch.no_grad():
                self.target_model.load_state_dict(self.policy_model.state_dict())        #？？ 是这样用吗


        # 梯度更新
        self.loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return self.loss



