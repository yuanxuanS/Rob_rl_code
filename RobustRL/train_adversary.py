import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from env import Environment
from graph import Graph_IM
from generate_node_feature import generate_node_feature, generate_edge_features
from hyperparam_model import hyper_model
from agent import DQAgent
from PPO_nature import PPOContinuousAgent

epoches = 100

T = 1
sub_T = 5
budget = 4  #
propagate_p = 0.7
q = 1     # willingness probability
cascade = None

dimensions = 3


# for t in range(T):
#     ## Environment definition ##
epsilon = 0.3

# nature_agent = DQAgent(graph, "")
batch_size = 2
update_target_steps = 2       # copy policy_model -> target model
main_lr = 1e-2

# nature
actor_lr = 1e-8
critic_lr = 1e-2
nature_lr = [actor_lr, critic_lr]
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
# plot
x_iter_t = []

y_cumulative_reward = []

# buffer_nature = []
# --------------- test agent, when training
for episode in range(1):
    # 初始化一个图结构:节点，连接关系, 节点特征。但传播参数不确定
    graph = Graph_IM(nodes=10, edges_p=0.5)
    # 环境初始化：graph, node features, state S, z
    env = Environment(T, sub_T, budget, graph, dimensions)      # T=1, one-step, adversary=bandit
    env.reset()

    # 初始化nature agent
    nature_observe_state = False
    nature_agent = PPOContinuousAgent(env, nature_lr, 'GAT_PPO', nature_observe_state, gamma, lmbda, eps, epochs)
    nature_agent.reset()


    # nature agent
    nature_state, _ = env.get_seed_state()
    z_action_pair_lst = nature_agent.act(nature_state)
    z_new = env.step_hyper(z_action_pair_lst)

    # main agent
    main_agent = DQAgent(env, main_lr, 'GAT_QN', epsilon, batch_size, env.sub_T, update_target_steps)
    main_agent.reset()
    cumul_reward = 0.

    sub_reward = []
    sub_loss = []
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
        sub_reward.append(reward)

        # get sample and update the main model, GAT
        loss = main_agent.update()
        sub_loss.append(loss)
        print(f"loss is {loss}")

    print(f"cumulative reward is {cumul_reward}")
        # plot
        # plt.plot(range(env.budget), sub_reward)
        # plt.title("reward per step")
        # plt.show()


    # nature agent
    nature_agent.remember(nature_state, z_action_pair_lst, -cumul_reward)
    # get a trajectory and update the nature model
    act_loss_nature, cri_loss_nature = nature_agent.update()
    print(f"actor loss {act_loss_nature} critic loss {cri_loss_nature}")
