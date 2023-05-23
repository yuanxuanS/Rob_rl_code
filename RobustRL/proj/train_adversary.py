import os

import numpy as np
import matplotlib.pyplot as plt
from env import Environment
from graph import Graph_IM
from RobustRL.proj.agent import DQAgent
from PPO_nature import PPOContinuousAgent
import sys
import time
sys.stdout = open(os.devnull, 'w')


def load_graph(graph_nbr_train, node_nbr):
    graph_dic = {}
    for graph_ in range(graph_nbr_train):
        seed = graph_
        graph_dic[graph_] = Graph_IM(nodes=node_nbr, edges_p=0.5, seed=seed)
        graph_dic[graph_].graph_name = str(graph_)

    print('train graphs in total: ', len(graph_dic))
    return graph_dic

def gener_node_features(node_nbr, node_dim, feat_nbr):
    n_feat_dic = {}
    for f in range(feat_nbr):
        seed = f
        np.random.seed(seed)
        n_feat_dic[f] = np.random.rand(node_nbr, node_dim)  # 0-1
    return n_feat_dic

def gener_z(node_dim, z_nbr):
    z_dic = {}
    for z_i in range(z_nbr):
        seed = z_i
        np.random.seed(seed)
        z_dic[z_i] = np.random.rand(1, 2 * node_dim)
    return z_dic



# train and env
epoches = 10
T = 3
budget = 4  #
propagate_p = 0.7
cascade = None
epsilon = 0.3
batch_size = 2
update_target_steps = 2       # copy policy_model -> target model
main_lr = 1e-3

# nature
actor_lr = 1e-5
critic_lr = 1e-3
nature_lr = [actor_lr, critic_lr]
PolicyDisName = "Beta"
PolicyNormName = "sigmoid"
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
canObserveState = False


# train
graph_nbr_train = 3
node_nbr = 200
graph_pool = load_graph(graph_nbr_train, node_nbr)
node_dim = 3
feat_nbr = 3
node_feat_pool = gener_node_features(node_nbr, node_dim, feat_nbr)
z_nbr = 3
z_pool = gener_z(node_dim, z_nbr)

env = Environment(graph_pool, node_feat_pool, z_pool, T, budget)      #
nature_agent = PPOContinuousAgent(graph_pool, node_feat_pool, z_pool, nature_lr, 'GAT_PPO', node_nbr, node_dim, PolicyDisName, PolicyNormName, canObserveState, gamma,
                                      lmbda, eps, epochs)
main_agent = DQAgent(graph_pool, node_feat_pool, z_pool, main_lr, 'GAT_QN', node_dim, epsilon, batch_size, update_target_steps)

st = time.time()
y_cumulative_reward = []
nature_critic_loss = []
nature_actor_loss = []
for g_id, graph in graph_pool.items():

    env.init_graph(g_id)
    nature_agent.init_graph(g_id)
    main_agent.init_graph(g_id)

    for ft_id, feat in node_feat_pool.items():
        env.init_n_feat(ft_id)
        nature_agent.init_n_feat(ft_id)
        main_agent.init_n_feat(ft_id)

        for hyper_id, hyper in z_pool.items():
            env.init_hyper(hyper_id)
            nature_agent.init_hyper(hyper_id)
            main_agent.init_hyper(hyper_id)


            # train
            episodes = 5
            for episode in range(episodes):       #  one-step, adversary is a bandit
                env.reset()
                nature_agent.reset()

                nature_state, _ = env.get_seed_state()
                z_action_pair_lst = nature_agent.act(nature_state)
                z_new = env.step_hyper(z_action_pair_lst)

                # main agent
                main_agent.reset()
                cumul_reward = 0.

                sub_reward = []
                sub_loss = []
                for i in range(env.budget * env.T):
                    print(f"---------- sub step {i}")
                    state, feasible_action = env.get_seed_state()     # [1, N]
                    action = main_agent.act(state, feasible_action)
                    print(f"action is {action} ")
                    next_state, reward, done = env.step_seed(i, action)

                    # add to buffer
                    sample = [state, action, reward, next_state, done, g_id, ft_id, hyper_id]
                    main_agent.remember(sample)

                    cumul_reward += reward
                    sub_reward.append(reward)

                    # get sample and update the main model, GAT
                    loss = main_agent.update(i)
                    sub_loss.append(loss)
                    print(f"loss is {loss}")

                y_cumulative_reward.append(cumul_reward)
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
                nature_critic_loss.append(cri_loss_nature.item())
                nature_actor_loss.append(act_loss_nature.item())

plt.figure()
plt.plot(range(len(y_cumulative_reward)), y_cumulative_reward)
plt.title(f"reward every episode, {node_nbr} nodes, {T} round, {budget} budget, time {time.time() - st}")
plt.savefig("reward")

plt.figure()
plt.plot(range(len(y_cumulative_reward)), nature_actor_loss)
plt.title("actor loss of nature agent every episode")
plt.savefig("act_loss")

plt.figure()
plt.plot(range(len(y_cumulative_reward)), nature_critic_loss)
plt.title("critic loss of nature agent every episode")
plt.savefig("cri_loss")

plt.show()