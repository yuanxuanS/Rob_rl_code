import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from env import Environment
from graph import Graph_IM
from generate_node_feature import generate_node_feature, generate_edge_features
from hyperparam_model import hyper_model
from agent import DQAgent
from PPO import PPOContinuousAgent
buffer = []

epoches = 100

T = 1
sub_T = 5
budget = 4  #
propagate_p = 0.7
q = 1     # willingness probability
cascade = None
# graph = nx.erdos_renyi_graph(n=3, p=0.5)

# graph = Graph_IM(nodes=10, edges_p=0.5)
# print(graph.graph)
# print(graph.graph[0])   # 图的第0个节点连接的边 {} 表示没有
# print([i for i in graph.graph[0]])   # 图的第1个节点的邻节点 和它的属性
# print(graph.graph[2])
# print(f"edges are {graph.edges}")
# ------------ test generate node feature module
dimensions = 3
# nodes_fea = generate_node_feature(graph, dimensions)
# edge_features = generate_edge_features(nodes_fea, graph)
# print(edge_features)
# z = np.random.rand(2*dimensions)        # 一维向量
# print(f"hyper param is {z}")
# hyper_model(edge_features, z)

# ------------ test environment
# env = Environment(T, budget, propagate_p, q, cascade, graph, dimensions)
# print(f"actions are {env.G.nodes}")       # list, node idx
# env.get_state()
# env.transition(1)  # 添加一个节点
# env.get_state()
# s1, r1 = env.step(2) # 再加一个节点，查看边际reward
# print("cur state is {} reward is {} ".format(s1, r1))
# print("total reward is {}".format(env.reward_total))
# s2, r2 = env.step(4)
# print("cur state is {} reward is {} ".format(s2, r2))
# print("total reward is {}".format(env.reward_total))
# print(graph.graph[1])

# for t in range(T):
#     ## Environment definition ##
epsilon = 0.3

# nature_agent = DQAgent(graph, "")
batch_size = 2
update_target_steps = 2       # copy policy_model -> target model
learning_rate = 1e-2

# plot
x_iter_t = []

y_cumulative_reward = []

# buffer_nature = []
# --------------- test agent, when training
for episode in range(1):
    # 初始化一个图结构:节点，连接关系, 节点特征。但传播参数不确定
    graph = Graph_IM(nodes=10, edges_p=0.5)
    # 环境初始化：graph, node features, state S, z
    env = Environment(T, sub_T, budget, graph, dimensions)
    env.reset()
    # 初始化nature agent
    # nature_agent = DQAgent(env.G, 'GAT_QN', epsilon, batch_size, env.T, update_t, main=False)
    # 修改超参z
    # z_cur = env.get_z_state()
    # z_next = nature_agent.act(z_cur, None)
    # env.step_hyper(z_next)
    for t in range(env.T):
        #
        # 初始化agent
        main_agent = DQAgent(env.G, learning_rate, env.node_feat_dimension, env.node_features, 'GAT_QN', epsilon, batch_size, env.sub_T, update_target_steps)
        main_agent.reset()
        cumul_reward = 0.
        # main agent
        y_reward = []
        y_loss = []
        for i in range(env.budget):
            print(f"---------- sub step {i}")
            state, feasible_action = env.get_seed_state()     # [1, N]
            action = main_agent.act(state, feasible_action)
            print(f"action is {action} and its type is {type(action)}")
            next_state, reward = env.step_seed(action)

            # add to buffer
            sample = [state, action, reward, next_state]
            main_agent.remember(sample)

            cumul_reward += reward
            y_reward.append(reward)

            # get sample and update the main model, GAT
            loss = main_agent.update()
            y_loss.append(loss)
            print(f"loss is {loss}")

        print(f"cumulative reward is {cumul_reward}")
        # plot
        plt.plot(range(env.budget), y_reward)
        plt.title("reward per step")
        plt.show()


        # add to buffer_nature
        # sample_nature = [z_cur, z_next, -cumul_reward]
        # nature_agent.remember(sample_nature)
        # get sample and update the nature model
        # loss_nature = nature_agent.update()
