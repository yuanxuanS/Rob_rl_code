from models import GAT, GAT_degree, GAT_degree2, GAT_origin, GAT_MLP, MLP
import torch
import numpy as np
import random
import copy
import seed
import logging
from graphviz import Digraph
from torchviz import make_dot, make_dot_from_trace
from torch.profiler import profile, record_function, ProfilerActivity
from er import ER
from per import PER
from utils import test_memory
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
    def __init__(self, graph_pool, node_num, node_feat_pool, hyper_pool, policy_name,
                 main_setting,
                 node_dim,
                 train_batch, use_cuda, device):

        self.use_cuda = use_cuda
        self.merge_z = main_setting["observe_z"]
        self.device = device

        # env
        self.graphs = graph_pool
        self.graph = None
        self.adj = None

        self.node_feat_pool = node_feat_pool
        self.node_features = None
        self.node_features_dims = node_dim

        self.hyper_pool = hyper_pool
        self.z = None

        self.nodes = node_num
        
        # policy
        self.policy_name = policy_name
        self.q_algor = main_setting["rl_algor"]
        self.method = main_setting["GAT_mtd"]       # if takes degree feature as input

        self.use_decay = main_setting["use_decay"]
        self.init_epsilon = main_setting["init_epsilon"]
        self.final_epsilon = main_setting["final_epsilon"]
        self.epsilon_decay_steps = main_setting["epsilon_decay_steps"]
        self.curr_epsilon = self.init_epsilon

        self.policy_model = None
        self.target_model = None

        # buffer
        from utils import process_config
        config = process_config(0, 0, 0)
        self.buffer_type = main_setting["buffer_type"]
        if self.buffer_type == "er":
            self.buffer = ER(main_setting["er"])
        elif self.buffer_type == "per_return":
            self.buffer = PER(**config["per_return"])
        elif self.buffer_type == "per_td":
            self.buffer = PER(**config["per_td"])
        elif self.buffer_type == "per_td_return":
            self.buffer = PER(**config["per_td_return"])
            self.td_reward_w = config["td_reward_weight"]

        # self.memory = []
        # self.buffer_max = 500  # equal to global iterations

        # train
        self.basic_batch_size = train_batch
        self.train_batch_size = train_batch  # 训练网络需要的样本数量

        self.gamma = main_setting["gamma"]
        self.criterion = torch.nn.MSELoss(reduction='mean')
        
        # self.copy_model_steps = update_target_steps
        self.target_model_last_update_t = main_setting["target_start_update_t"]
        self.target_update_interval = main_setting["target_update_interval"]
        self.lr = main_setting["lr"]

        # log
        self.path = None
        self.test_mem =False

        if self.policy_name == 'GAT_QN':

            alpha = main_setting["alpha"]  # leakyReLU的alpha
            nhead = main_setting["nheads"]

            layer_tp = (main_setting["GAT_atten_layer"], main_setting["GAT_out_atten_layer"])
            hid_dim_tp = (main_setting["GAT_hid_dim"], main_setting["GAT_out_hid_dim"])
            hid_s_dim_tp = (main_setting["GAT_s_hid_dim"], main_setting["GAT_s_out_hid_dim"])

            self.nnmodel = main_setting["nnVersion"]
            logging.info(f"nnmodel {self.nnmodel}")
            if self.nnmodel == "v2" or self.nnmodel == "v3":
                out_nhid_tp = hid_dim_tp[-1]
                # assert out_nhid_tp[-1] == 1, "out feature of agent model must be one"
                # logging.debug(f"nnmodel {self.nnmodel}")
                self.policy_model = GAT_degree(self.nnmodel, self.nodes, layer_tp, nfeat=self.node_features_dims,
                                               nhid_tuple=hid_dim_tp,
                                               nfeat_s=self.nodes, nhid_s_tuple=hid_s_dim_tp,
                                               alpha=alpha, nheads=nhead, mergeZ=self.merge_z, mergeState=True,
                                               use_cuda=self.use_cuda, device=self.device, method=self.method)
                self.target_model = GAT_degree(self.nnmodel, self.nodes, layer_tp, nfeat=self.node_features_dims,
                                               nhid_tuple=hid_dim_tp,
                                               nfeat_s=self.nodes, nhid_s_tuple=hid_s_dim_tp,
                                               alpha=alpha, nheads=nhead, mergeZ=self.merge_z, mergeState=True,
                                               use_cuda=self.use_cuda, device=self.device, method=self.method)
            elif self.nnmodel == "v1":
                # GAT输出维度不需要是1
                self.policy_model = GAT_degree2(layer_tp, nfeat=self.node_features_dims, nhid_tuple=hid_dim_tp,
                                                alpha=alpha, nheads=nhead, mergeZ=self.merge_z, mergeState=True,
                                                use_cuda=self.use_cuda, device=self.device, method=self.method)
                self.target_model = GAT_degree2(layer_tp, nfeat=self.node_features_dims, nhid_tuple=hid_dim_tp,
                                                alpha=alpha, nheads=nhead, mergeZ=self.merge_z, mergeState=True,
                                                use_cuda=self.use_cuda, device=self.device, method=self.method)
            elif self.nnmodel == "v4":
                self.policy_model = GAT(layer_tp, self.node_features_dims + self.nodes, hid_dim_tp,
                                        alpha, nhead, self.merge_z, True,
                                        self.use_cuda, self.device, method=self.method)
                self.target_model = GAT(layer_tp, self.node_features_dims + self.nodes, hid_dim_tp,
                                        alpha, nhead, self.merge_z, True,
                                        self.use_cuda, self.device, method=self.method)
            elif self.nnmodel == "v01":
                '''
                mlp_layer = 3
                mlp_hid = 10
                self.policy_model = GAT_MLP(mlp_layer, mlp_hid, layer_tp, self.node_features_dims,
                                    hid_dim_tp, alpha, nhead, self.merge_z, True, self.use_cuda, self.device, self.method)
                self.target_model = GAT_MLP(mlp_layer, mlp_hid, layer_tp, self.node_features_dims,
                                    hid_dim_tp, alpha, nhead, self.merge_z, True, self.use_cuda, self.device, self.method)
                '''
                self.policy_model = GAT(self.nodes, layer_tp, self.node_features_dims,
                                    hid_dim_tp, alpha, nhead, self.merge_z, True, self.use_cuda, self.device, self.method)
                self.target_model = GAT(self.nodes, layer_tp, self.node_features_dims,
                                    hid_dim_tp, alpha, nhead, self.merge_z, True, self.use_cuda, self.device, self.method)
                
            elif self.nnmodel == "v0":
                # nhid = 8, out-hid =32
                self.policy_model = GAT_origin(1, 8, 0, 32, self.nodes, 0, alpha=alpha, nheads=nhead,
                        layer_type="default")
                self.target_model = GAT_origin(1, 8, 0, 32, self.nodes, 0, alpha=alpha,
                                               nheads=nhead, layer_type="default")
            elif self.nnmodel == "v5":
                layers = 3
                self.policy_model = MLP(layers, 16, 3, 1)
                self.target_model = MLP(layers, 16, 3, 1)
                

            if self.use_cuda:
                self.policy_model.to(self.device)
                self.target_model.to(self.device)

            logging.info(f"model parameters :\n")
            for name, param in self.policy_model.named_parameters():
                logging.info(f"this layer: {name}, required grad: {param.requires_grad}")
                
        
            self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)

        elif self.policy_name == "random":
            self.policy_model = None
            self.target_model = None

        # test
        self.print_tag = "DQN Agent---"

    def init_graph(self, g_id):
        '''
        initial graph in every episode
        '''
        self.graph = self.graphs[g_id]  # a graph, Graph_IM instance
        self.adj = torch.Tensor(self.graph.adj_matrix)
        self.s_mat = self.graph.adm

    def init_n_feat(self, ft_id):
        '''
        initial node feature in every episode
        '''
        self.node_features = torch.Tensor(self.node_feat_pool[ft_id])

    def init_hyper(self, hyper_id):
        '''
        initial hyper parameter in every episode
        '''
        self.z = torch.Tensor(self.hyper_pool[hyper_id])

    def reset(self):
        '''
        in every episode
        '''
        # self.global_step = 1
        pass
        # print(f"{self.print_tag} agent reset done!")

    def epsilon_decay(self, init_v: float, final_v: float, step_t: int, decay_step: int):
        assert 0 < final_v <= 1, ValueError('Value Error')
        assert step_t >= 0, ValueError('Value Error')
        assert decay_step > 0, ValueError('Decay Value Error')
        
        if step_t >= decay_step:
            return final_v
        return step_t * ((final_v - init_v)/float(decay_step)) + init_v


    @torch.no_grad()
    def act(self, observation, feasible_action, glb_steps, mode):
        # policy
        if self.policy_name == 'GAT_QN':
            if self.use_decay > 0:
                self.curr_epsilon = self.epsilon_decay(self.init_epsilon, self.final_epsilon, glb_steps, self.epsilon_decay_steps)
            else:
                self.curr_epsilon = self.init_epsilon
            logging.info(f"epsilon is {self.curr_epsilon}")

            if self.curr_epsilon > np.random.rand() and mode != "valid":
                action = np.random.choice(feasible_action)
                # logging.info(f"action is {action}")
            else:
                
                if self.test_mem:
                    test_memory()

                input_node_feat = self.node_features

                if not isinstance(input_node_feat, torch.Tensor):
                    input_node_feat = torch.Tensor(input_node_feat)
                if not isinstance(self.s_mat, torch.Tensor):
                    self.s_mat = torch.Tensor(self.s_mat)

                if self.test_mem:
                    test_memory()
                '''
                if self.nnmodel == "v4":
                    input_node_feat = torch.concat((input_node_feat, self.s_mat), 1)
                if self.nnmodel == "v0":
                    input_node_feat = observation.T
                    logging.info(f"input node feature size of v0 is {input_node_feat}")
                '''

                if self.test_mem:
                    test_memory()

                # with profile(activities=[ProfilerActivity.CPU],
                #             profile_memory=True, record_shapes=True) as prof:
                if self.nnmodel == "v01":
                    q_a = self.policy_model(input_node_feat.to(self.device), self.adj.to(self.device),
                                            torch.Tensor(observation).to(self.device), None,
                                            z=self.z.to(self.device))
                    # logging.info(f"v01 q is {q_a.T}")
                '''
                elif self.nnmodel == "v0":
                    input_node_feat = input_node_feat[None, ...]
                    if not isinstance(input_node_feat, torch.Tensor):
                        input_node_feat = torch.Tensor(input_node_feat)
                    q_a = self.policy_model(input_node_feat.to(self.device), self.adj.to(self.device))
                elif self.nnmodel == "v5":
                    q_a = self.policy_model(input_node_feat.to(self.device))
                else:

                    q_a = self.policy_model(input_node_feat.to(self.device), self.adj.to(self.device),
                                            torch.Tensor(observation).to(self.device), self.s_mat,
                                            z=self.z.to(self.device))
                '''
                logging.info(f" get action done")
                if self.test_mem:
                    test_memory()
                # print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))

                
                if self.use_cuda:
                    q_a = q_a.cpu()
                '''
                if self.nnmodel == "v0":
                    q_a = torch.squeeze(q_a, dim=0)
                '''
                if self.test_mem:
                    test_memory()

                infeasible_action = [k for k in range(self.nodes) if k not in feasible_action]
                # print(f"{self.print_tag} infeasible action is {infeasible_action}")

                q_a[infeasible_action] = -9e15

                # logging.debug(f"get model params")
                # for name, param in self.policy_model.named_parameters():
                #     logging.debug(f"this layer: {name}, params: {param}")

                # logging.info(f"act, final q_a is {q_a.T}")
                action = q_a.argmax()

                if self.test_mem:
                    test_memory()

        elif self.policy_name == "random":
            action = np.random.choice(feasible_action)

        if not isinstance(action, int):
            action = int(action)
        return action

    def remember(self, sample_lst):
        '''

        :param sample_lst: [state, action, reward, next_state, feasible_action, done, g_id, ft_id, hyper_id]
        :return:
        '''
        @torch.no_grad()
        def calculate_td_error(obs, action, reward, next_obs, feasible_action, done, g_id, ft_id, hyper_id):
            #using is the absolute error, not square error
            graph = self.graphs[g_id]
            adj = torch.Tensor(graph.adj_matrix)
            node_feature = torch.Tensor(self.node_feat_pool[ft_id])
            hyper = torch.Tensor(self.hyper_pool[hyper_id])
            infeasible_action = [k for k in range(self.graph.node) if k not in feasible_action]

            # logging.info(f"state\n{obs}\naction\n{action}\nreward\n{reward}\nnext_state\n{next_obs} \
            #     feasible_\n {feasible_action} \n gid_\n {g_id} \n ftid_ {ft_id} \n hyid_ {hyper_id}")
        
            if self.q_algor == "DDQN":
                qvalues = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                            (torch.Tensor(obs)).to(self.device), torch.Tensor(self.s_mat).to(self.device),
                                            z=hyper.to(self.device))
                # logging.info(f"qvalues are {qvalues}")
                prediction = qvalues[action]
                # logging.info(f"prediction are {prediction}")

                target_val = self.target_model(node_feature.to(self.device), adj.to(self.device),
                                        torch.Tensor(next_obs).to(self.device), torch.Tensor(self.s_mat).to(self.device),
                                        z=hyper.to(self.device))
                # logging.info(f"target values are {target_val}")

                q_a_tmp = qvalues.clone()  # 深拷贝，不能改变原值
                q_a_tmp[infeasible_action] = -9e15
                # logging.info(f"q a, after mask: \n{q_a_tmp}")
                policy_max_action = q_a_tmp.argmax()
                # logging.info(f"policy max action is {policy_max_action}")
                # logging.info(f"q_target[max] is {target_val[policy_max_action]}")
                target = reward + (1 - done) * self.gamma * target_val[policy_max_action]


            priority_value = abs(target - prediction).item()
            # logging.info(f"target is {target} , prediction is {prediction}, prior is {priority_value}")
            return priority_value
        
        if self.buffer_type == "er":
            priority_value = 0
        else:
            state, action, reward, next_state, feasible_action, done, g_id, ft_id, hyper_id = sample_lst
            if self.buffer_type == "per_td":
                priority_value = calculate_td_error(state, action, reward, next_state, feasible_action,
                                done, g_id, ft_id, hyper_id)
            elif self.buffer_type == "per_return":
                priority_value = reward
            elif self.buffer_type == "per_td_return":
                td_error = calculate_td_error(state, action, reward, next_state, feasible_action,
                                done, g_id, ft_id, hyper_id)
                priority_value = abs(td_error) + self.td_reward_w * reward
                # logging.info(f"td error is {td_error}, w {self.td_reward_w}, reward {reward}, pri {priority_value}")
        self.buffer.append(sample_lst, priority_value)

    def get_sample(self):

        # self.train_batch_size = int((1 + len(self.memory) / self.buffer_max) * self.basic_batch_size)
        self.train_batch_size = self.basic_batch_size

        if len(self.buffer) > self.train_batch_size:

            batch, idxs, is_weight = self.buffer.sample(self.train_batch_size)
            # logging.info(f"{self.print_tag} batch is {batch}")
            # print(f" zip is {list(zip(*batch))}")
            state_batch = list(torch.Tensor(list(zip(*batch))[0]))
            # print(f"state batch is {state_batch}")
            action_batch = list(list(zip(*batch))[1])
            reward_batch = list(list(zip(*batch))[2])
            next_state_batch = list(torch.Tensor(list(zip(*batch))[3]))
            feasible_batch = list(list(zip(*batch))[4])
            done_batch = list(list(zip(*batch))[5])
            gid_batch = list(list(zip(*batch))[6])
            ftid_batch = list(list(zip(*batch))[7])
            hyid_batch = list(list(zip(*batch))[8])
        else:
            batch = []
            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            feasible_batch = []
            done_batch = []
            gid_batch = []
            ftid_batch = []
            hyid_batch= []
            idxs = []
            is_weight = []
        return batch, idxs, is_weight
        # return state_batch, action_batch, reward_batch, next_state_batch, feasible_batch, done_batch, gid_batch, ftid_batch, hyid_batch

    def update(self, global_step):
        
        if self.test_mem:
            test_memory()   
        
        batch, idxs, is_weight = self.get_sample()
        # logging.info(f"idxs is {idxs}\n is_weight is {is_weight}")
        if not batch:
            # print(f"{self.print_tag} no enough sample and no update")
            return 0.
        else:
            logging.info(f"batch length is {len(batch)}")
        # check gradients
        # self.hs = []
        # for name, module in self.policy_model.named_children():
        #     logging.debug(f"add hook")
        #     h = module.register_backward_hook(self.backward_hook)
        #     self.hs.append(h)


        
        i = 0       # record iter of transition time, to concat target and q
        # losses = 0.

        if self.test_mem:
            logging.debug(f"before batches")
            test_memory()

        for transition in batch:

            
            state, action, reward, next_state, feasible_a, done, g_id, ft_id, hyper_id = transition
            # logging.info(f"state\n{state}\naction\n{action}\nreward\n{reward}\nnext_state\n{next_state} \
            #     feasible_\n {feasible_a} \n gid_\n {g_id} \n ftid_ {ft_id} \n hyid_ {hyper_id}")
        
            # 用目标网络计算目标值y
            graph = self.graphs[g_id]
            adj = torch.Tensor(graph.adj_matrix)
            node_feature = torch.Tensor(self.node_feat_pool[ft_id])
            hyper = torch.Tensor(self.hyper_pool[hyper_id])

            if self.test_mem:
                logging.debug(f"p1")
                test_memory()

            if not isinstance(self.s_mat, torch.Tensor):
                self.s_mat = torch.Tensor(self.s_mat)
            
            '''
            if self.nnmodel == "v0":
                node_feature = state.T
                # logging.debug(f" node feature size of v0 is {node_feature.size()}")
            '''
            if self.test_mem:
                logging.debug(f"p2")
                test_memory()

            if self.nnmodel == "v0":
                if not isinstance(node_feature, torch.Tensor):
                    node_feature = torch.Tensor(node_feature)
                node_feature = node_feature[None, ...]
                q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device))
            else:
                if self.nnmodel == "v4":
                    logging.info(f"ndoe feature size {node_feature.size()}, s_mat size {self.s_mat.size()}")
                    node_feature = torch.concat((node_feature, self.s_mat), 1)

                    q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                            torch.Tensor(state).to(self.device), None,
                                            z=hyper.to(self.device))
                elif self.nnmodel == "v5":
                    q_a = self.policy_model(node_feature.to(self.device))
                else:
                    q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                            torch.Tensor(state).to(self.device), None,
                                            z=hyper.to(self.device))
            
            if self.test_mem:
                logging.debug(f"p3")
                test_memory()
            '''
            if self.nnmodel == "v0": # 该网络输出第一个维度是batchsizw
                
                q_a = torch.squeeze(q_a, dim=0)
                logging.info(f"v01 q size is {q_a.size()}")
            '''

            q = q_a[action]

            if self.test_mem:
                logging.debug(f"p4")
                test_memory()

            if self.nnmodel == "v0":
                next_s = next_state.T
                if not isinstance(next_s, torch.Tensor):
                    next_s = torch.Tensor(next_s)
                next_s = next_s[None, ...]
                q_target = self.target_model(next_s.to(self.device), adj.to(self.device))
            else:
                if self.nnmodel == "v01":
                    q_target = self.target_model(node_feature.to(self.device), adj.to(self.device),
                                            torch.Tensor(next_state).to(self.device), None,
                                            z=hyper.to(self.device))
                elif self.nnmodel == "v5":
                    q_target = self.target_model(node_feature.to(self.device))
                else:
                    q_target = self.target_model(node_feature.to(self.device), adj.to(self.device),
                                        torch.Tensor(next_state).to(self.device), self.s_mat.to(self.device),
                                        z=hyper.to(self.device))
            
            if self.test_mem:
                logging.debug(f"p5")
                test_memory()
            '''
            if self.nnmodel == "v0":
                q_target = torch.squeeze(q_target, dim=0)
                logging.info(f"v0 q target size is {q_target.size()}")
            '''
            # logging.debug(f"target nn is {q_target}")
            infeasible_action = [k for k in range(self.nodes) if k not in feasible_a]
            # print(f"{self.print_tag} infeasible action is {infeasible_action}")

            if self.test_mem:
                logging.debug(f"p6")
                test_memory()

            if self.q_algor == "DQN":
                q_target[infeasible_action] = -9e15
                target = reward + (1 - done) * self.gamma * q_target.detach().max()
            elif self.q_algor == "DDQN":
                # logging.debug(f"q target, max value is {q_target.max()}")
                # logging.debug(f"q a, before mask: \n{q_a}")
                q_a_tmp = q_a.clone()  # 深拷贝，不能改变原值
                q_a_tmp[infeasible_action] = -9e15
                # logging.debug(f"q a, after mask: \n{q_a}")
                policy_max_action = q_a_tmp.argmax()
                # logging.debug(f"policy max action is {policy_max_action}")
                # logging.debug(f"q_target[max] is {q_target[policy_max_action]}")
                target = reward + (1 - done) * self.gamma * q_target[policy_max_action].detach()
                # logging.debug(f"target value is {target}")
            
            if self.test_mem:
                logging.debug(f"p7")
                test_memory()

            if not isinstance(target, torch.Tensor):
                target = torch.Tensor([target])
            # print(f"{self.print_tag} calculated target q is {target}")
            # 用行为网络计算当前值q

            # g_b = make_dot(q_a)
            # g_b.render(
            #     filename="all q value",
            #     directory=self.path + "/logdir",
            #     format="png"
            # )

            # h = q.register_hook(self.hook)
            # self.hs.append(h)

            if self.buffer_type != "er":
                # update priority
                idx = idxs[i]
                if self.buffer_type == "per_td":
                    td_error = abs((q.detach() - target).data)
                    self.buffer.update(idx, td_error)

                elif self.buffer_type == "per_return":
                    self.buffer.update(idx, reward)

                elif self.buffer_type == "per_td_return":
                    td_error = abs((q.detach() - target).data)
                    priority = td_error + self.td_reward_w * reward
                    self.buffer.update(idx, priority)
                    # logging.info(f"idx is {idx}, priority is {priority}")

            if i == 0:

                # logging.debug(f"q is {q}")
                # logging.debug(f"target is {target}")
                qs = torch.Tensor([q])
                ts = torch.Tensor([target])
            else:
                qs = torch.concat((qs, q), 0)
                ts = torch.concat((ts, target), 0)
            # loss = self.criterion(q, target)
            if self.test_mem:
                logging.debug(f"p8")
                test_memory()
            i += 1
        # losses_a = make_dot(losses)
        # losses_a.render(
        #     filename="losses",
        #     directory=self.path + "/logdir",
        #     format="png"
        # )
        # logging.debug(f"q batch is {qs}")
        # logging.debug(f"targets is {ts}")
        # loss = torch.mean(torch.tensor(losses, requires_grad=True))
        # loss = losses / len(batch)
        # loss_a = make_dot(loss)
        # loss_a.render(
        #     filename="loss",
        #     directory=self.path + "/logdir",
        #     format="png"
        # )
        # h = loss.register_hook(self.hook)
        # self.hs.append(h)
        # logging.debug(f" loss , requires_grad {loss.requires_grad}, grad {loss.grad}")
        # print(f"{self.print_tag} update losses are {losses} and loss is {loss}")

        # torch.autograd.set_detect_anomaly(True)
        # 梯度更新
        # logging.info(f"this update: q is {qs}, target is {ts}")
        if self.test_mem:
            logging.debug(f"after batches")
            test_memory()

        if self.buffer_type != "er":
            loss = torch.mean(torch.Tensor(is_weight) * (qs - ts)**2)
        else:
            loss = self.criterion(qs, ts)

        # logging.info(f"qs - ts : {qs-ts}")
        # logging.info(f"qs - ts ^2: {(qs-ts)**2}")
        # logging.info(f"* weight: {torch.Tensor(is_weight)* (qs-ts)**2}")
        # logging.info(f"loss is {loss}")
        self.loss = loss.item()
        self.optimizer.zero_grad()
        # logging.info(f"before backward")
        # for name, param in self.policy_model.named_parameters():
        #     logging.info(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")

        loss.backward()

        if self.test_mem:
            test_memory()
        # logging.info(f"\nafter backward")

        # torch.set_printoptions(profile="full")
        # for name, param in self.policy_model.named_parameters():
        #     logging.info(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")
        # torch.set_printoptions(profile="default")

        self.optimizer.step()

        if self.test_mem:
            test_memory()

        # for h in self.hs:
        #     h.remove()

        # 每 C step，更新目标网络 = 当前的行为网络
        if global_step - self.target_model_last_update_t > self.target_update_interval:
            logging.info(f"global step is {global_step}, self.copy m st: {self.target_update_interval},  reload target model from policy model ")
            with torch.no_grad():
                self.target_model.load_state_dict(self.policy_model.state_dict())  #
            # logging.debug(f"after reload, target model params: {self.target_model.state_dict()}")
            self.target_model_last_update_t = global_step
        if self.test_mem:
            test_memory()
        del qs, ts, loss
        return self.loss

    def update2(self, i):
        # 采样batch更新policy_model
        # 从memory中采样
        if self.test_mem:
            test_memory()   
        
        state_batch, action_batch, reward_batch, next_state_batch, feasible_batch, done_batch, gid_batch, ftid_batch, hyid_batch = self.get_sample()
        if not state_batch:
            # print(f"{self.print_tag} no enough sample and no update")
            return 0.
        else:
            logging.debug(f"batch length is {len(state_batch)}")
            # logging.debug(f"state_batch\n{state_batch}\naction_batch\n{action_batch}\nreward_batch\n{reward_batch}\nnext_state_batch\n{next_state_batch} \
            #     feasible_batch\n {feasible_batch} \n gid_batch\n {gid_batch} \n ftid_batch {ftid_batch} \n hyid_batch {hyid_batch}")
        # check gradients
        self.hs = []
        # for name, module in self.policy_model.named_children():
        #     logging.debug(f"add hook")
        #     h = module.register_backward_hook(self.backward_hook)
        #     self.hs.append(h)

        # losses = torch.tensor(0.)
        
        
        i = 0
        losses = 0.
        if self.test_mem:
            logging.debug(f"before batches")
            test_memory()

        if self.nnmodel == "v0":

            adjs = []       # list of tensors
            for gid in gid_batch:
                graph = self.graphs[gid]
                adj = torch.Tensor(graph.adj_matrix)
                adjs.append(adj)

            node_feature = [state.T for state in state_batch]
            logging.debug(f"state batchm tensor\n {node_feature}")

            q_a = self.policy_model(node_feature, adjs)

            q = q_a[range(self.train_batch_size), action_batch, :]
            # logging.info(f"q \n {q}")

            next_s = [ns.T for ns in next_state_batch]
            q_target = self.target_model(next_s, adjs)
            # logging.debug(f"target {q_target}")


            # infeasible_as = [ for feasible_a in feasible_batch]
            
            logging.debug(f"q_target after infea {q_target}")

            if self.q_algor == "DQN":
                i = -1
                for feasible_a in feasible_batch:
                    i += 1
                    infeasible_action = [k for k in range(self.graph.node) if k not in feasible_a]
                    q_target[i, infeasible_action, :] = -9e15
                # logging.info(f"size of q target max is{q_target.detach().max(dim=1).values.size()}")  # [bs, 1]
                target = torch.Tensor(reward_batch)[..., None] + (1 - torch.Tensor(done_batch)[..., None]) * self.gamma * q_target.detach().max(dim=1).values
                logging.debug(f"size if target {target.size()}")
            
            elif self.q_algor == "DDQN":
                # logging.debug(f"q target, max value is {q_target.max()}")
                # logging.debug(f"q a, before mask: \n{q_a}")
                q_a_tmp = q_a.clone()  # 深拷贝，不能改变原值
                i = -1
                for feasible_a in feasible_batch:
                    i += 1
                    infeasible_action = [k for k in range(self.graph.node) if k not in feasible_a]
                    # logging.debug(f"infea a {infeasible_action}")
                    q_a_tmp[i, infeasible_action, :] = -9e15
                # logging.debug(f"q a, after mask: \n{q_a_tmp}")
                policy_max_action = q_a_tmp.argmax(dim=1).squeeze(1).tolist()
                # logging.debug(f"policy max action is {policy_max_action}")       # 1d list
                # logging.debug(f"q_target[max] is { q_target[range(self.basic_batch_size), policy_max_action, :].detach()}")      # [bs, 1]
                # logging.debug(f"done  is { (1 - torch.Tensor(done_batch)[..., None])}") # [bs, 1]
                # logging.debug(f"torch.Tensor(reward_batch)[..., None, None] is { torch.Tensor(reward_batch)[..., None]}") # [bs, 1]
                target = torch.Tensor(reward_batch)[..., None] + (1 - torch.Tensor(done_batch)[..., None]) * self.gamma * q_target[range(self.basic_batch_size), policy_max_action, :].detach()
                # logging.debug(f"target value is {target}") # [bs, 1]
                
        loss = self.criterion(q, target)
        logging.info(f"loss is {loss}")
        self.loss = loss.item()
        self.optimizer.zero_grad()
        # logging.debug(f"before backward")
        # for name, param in self.policy_model.named_parameters():
        #     logging.debug(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")

        loss.backward()

        if self.test_mem:
            test_memory()
        logging.debug(f"\nafter backward")
        for name, param in self.policy_model.named_parameters():
            logging.debug(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")
            # pass
        self.optimizer.step()

        if self.test_mem:
            test_memory()

        for h in self.hs:
            h.remove()

        # 每 C step，更新目标网络 = 当前的行为网络
        if i % self.copy_model_steps == 0:
            # logging.debug(f"reload target model, policy model params: {self.policy_model.state_dict()}\n target model param: {self.target_model.state_dict()}")
            with torch.no_grad():
                self.target_model.load_state_dict(self.policy_model.state_dict())  #
            # logging.debug(f"after reload, target model params: {self.target_model.state_dict()}")

        if self.test_mem:
            test_memory()
            '''
        for transition in batch:

            
            state, action, reward, next_state, feasible_a, done, g_id, ft_id, hyper_id = transition
            # 用目标网络计算目标值y
            graph = self.graphs[g_id]
            adj = torch.Tensor(graph.adj_matrix)
            node_feature = torch.Tensor(self.node_feat_pool[ft_id])
            hyper = torch.Tensor(self.hyper_pool[hyper_id])

            if self.test_mem:
                logging.debug(f"p1")
                test_memory()

            if not isinstance(self.s_mat, torch.Tensor):
                self.s_mat = torch.Tensor(self.s_mat)
            
            
                # logging.debug(f" node feature size of v0 is {node_feature.size()}")
            
            if self.test_mem:
                logging.debug(f"p2")
                test_memory()

            if self.nnmodel == "v0":
                
                node_feature = node_feature[None, ...]
                
            else:
                if self.nnmodel == "v4":
                    logging.info(f"ndoe feature size {node_feature.size()}, s_mat size {self.s_mat.size()}")
                    node_feature = torch.concat((node_feature, self.s_mat), 1)

                    q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                            torch.Tensor(state).to(self.device), None,
                                            z=hyper.to(self.device))
                elif self.nnmodel == "v5":
                    q_a = self.policy_model(node_feature.to(self.device))
                else:
                    q_a = self.policy_model(node_feature.to(self.device), adj.to(self.device),
                                            torch.Tensor(state).to(self.device), self.s_mat.to(self.device),
                                            z=hyper.to(self.device))
            
            if self.test_mem:
                logging.debug(f"p3")
                test_memory()

            if self.nnmodel == "v0": # 该网络输出第一个维度是batchsizw
                
                q_a = torch.squeeze(q_a, dim=0)
                logging.info(f"v01 q size is {q_a.size()}")
            

            q = q_a[action]

            if self.test_mem:
                logging.debug(f"p4")
                test_memory()

            if self.nnmodel == "v0":
                next_s = next_state.T
                if not isinstance(next_s, torch.Tensor):
                    next_s = torch.Tensor(next_s)
                next_s = next_s[None, ...]
                q_target = self.target_model(next_s.to(self.device), adj.to(self.device))
            else:
                if self.nnmodel == "v01":
                    q_target = self.target_model(node_feature.to(self.device), self.adj.to(self.device),
                                            torch.Tensor(next_state).to(self.device), None,
                                            z=hyper.to(self.device))
                elif self.nnmodel == "v5":
                    q_target = self.target_model(node_feature.to(self.device))
                else:
                    q_target = self.target_model(node_feature.to(self.device), self.adj.to(self.device),
                                        torch.Tensor(next_state).to(self.device), self.s_mat.to(self.device),
                                        z=hyper.to(self.device))
            
            if self.test_mem:
                logging.debug(f"p5")
                test_memory()
            
            if self.nnmodel == "v0":
                q_target = torch.squeeze(q_target, dim=0)
                logging.info(f"v0 q target size is {q_target.size()}")
            
            # logging.debug(f"target nn is {q_target}")
            infeasible_action = [k for k in range(self.graph.node) if k not in feasible_a]
            # print(f"{self.print_tag} infeasible action is {infeasible_action}")

            if self.test_mem:
                logging.debug(f"p6")
                test_memory()

            if self.q_algor == "DQN":
                q_target[infeasible_action] = -9e15
                target = reward + (1 - done) * self.gamma * q_target.detach().max()
            elif self.q_algor == "DDQN":
                # logging.debug(f"q target, max value is {q_target.max()}")
                # logging.debug(f"q a, before mask: \n{q_a}")
                q_a_tmp = q_a.clone()  # 深拷贝，不能改变原值
                q_a_tmp[infeasible_action] = -9e15
                # logging.debug(f"q a, after mask: \n{q_a}")
                policy_max_action = q_a_tmp.argmax()
                # logging.debug(f"policy max action is {policy_max_action}")
                # logging.debug(f"q_target[max] is {q_target[policy_max_action]}")
                target = reward + (1 - done) * self.gamma * q_target[policy_max_action].detach()
                # logging.debug(f"target value is {target}")
            
            if self.test_mem:
                logging.debug(f"p7")
                test_memory()

            if not isinstance(target, torch.Tensor):
                target = torch.Tensor([target])
            # print(f"{self.print_tag} calculated target q is {target}")
            # 用行为网络计算当前值q

            # g_b = make_dot(q_a)
            # g_b.render(
            #     filename="all q value",
            #     directory=self.path + "/logdir",
            #     format="png"
            # )

            # h = q.register_hook(self.hook)
            # self.hs.append(h)
            if i == 0:

                # logging.debug(f"q is {q}")
                # logging.debug(f"target is {target}")
                qs = torch.Tensor([q])
                ts = torch.Tensor([target])
            else:
                qs = torch.concat((qs, q), 0)
                ts = torch.concat((ts, target), 0)
            # loss = self.criterion(q, target)
            if self.test_mem:
                logging.debug(f"p8")
                test_memory()
            i += 1
        # losses_a = make_dot(losses)
        # losses_a.render(
        #     filename="losses",
        #     directory=self.path + "/logdir",
        #     format="png"
        # )
        # logging.debug(f"q batch is {qs}")
        # logging.debug(f"targets is {ts}")
        # loss = torch.mean(torch.tensor(losses, requires_grad=True))
        # loss = losses / len(batch)
        # loss_a = make_dot(loss)
        # loss_a.render(
        #     filename="loss",
        #     directory=self.path + "/logdir",
        #     format="png"
        # )
        # h = loss.register_hook(self.hook)
        # self.hs.append(h)
        # logging.debug(f" loss , requires_grad {loss.requires_grad}, grad {loss.grad}")
        # print(f"{self.print_tag} update losses are {losses} and loss is {loss}")

        # torch.autograd.set_detect_anomaly(True)
        # 梯度更新
        # logging.debug(f"this update: q is {qs}, target is {ts}")
        if self.test_mem:
            logging.debug(f"after batches")
            test_memory()

        loss = self.criterion(qs, ts)
        logging.info(f"loss is {loss}")
        self.loss = loss.item()
        self.optimizer.zero_grad()
        # logging.debug(f"before backward")
        # for name, param in self.policy_model.named_parameters():
        #     logging.debug(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")

        loss.backward()

        if self.test_mem:
            test_memory()
        logging.debug(f"\nafter backward")
        for name, param in self.policy_model.named_parameters():
            logging.debug(f"this layer: {name}, required grad: {param.requires_grad}, gradients: {param.grad}")
            # pass
        self.optimizer.step()

                '''
        del q, target, loss

        return self.loss




