import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from env import Environment
from graph import Graph_IM
from generate_node_feature import generate_node_feature, generate_edge_features
from hyperparam_model import hyper_model
from agent import DQAgent
from PPO_nature import PPOContinuousAgent
import sys

sys.stdout = open(os.devnull, 'w')
epoches = 100

T = 1
budget = 4  #
propagate_p = 0.7
cascade = None

node_dim = 3

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
epochs = 100
eps = 0.2
# plot


y_cumulative_reward = []
nature_critic_loss = []
nature_actor_loss = []

# one graph with multi episodes
graph = Graph_IM(nodes=10, edges_p=0.5)
env = Environment(T, budget, graph, node_dim)      # T=1, one-step, adversary is a bandit
env.init()

canObserveState = False
nature_agent = PPOContinuousAgent(env, nature_lr, 'GAT_PPO', PolicyDisName, PolicyNormName, canObserveState, gamma,
                                  lmbda, eps, epochs)

main_agent = DQAgent(env, main_lr, 'GAT_QN', epsilon, batch_size, update_target_steps)

episodes = 10
for episode in range(episodes):
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
    for i in range(env.budget):
        print(f"---------- sub step {i}")
        state, feasible_action = env.get_seed_state()     # [1, N]
        action = main_agent.act(state, feasible_action)
        print(f"action is {action} ")
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
plt.plot(range(episodes), y_cumulative_reward)
plt.title("reward every episode")
plt.savefig("reward")

plt.figure()
plt.plot(range(episodes), nature_actor_loss)
plt.title("actor loss of nature agent every episode")
plt.savefig("act_loss")

plt.figure()
plt.plot(range(episodes), nature_critic_loss)
plt.title("critic loss of nature agent every episode")
plt.savefig("cri_loss")

plt.show()