import random
import numpy as np
class Runner:
    def __init__(self, environment, env_setting, agent, main_setting, nature,
                 training_setting, device_setting, writer):
        self.environment = environment
        self.env_setting = env_setting
        self.agent = agent
        self.main_setting = main_setting
        self.nature = nature

        self.training_setting = training_setting
        self.device_setting = device_setting
        self.writer = writer


        self.nature_critic_loss = []
        self.nature_actor_loss = []
        self.main_loss = []

        self.global_iter = 0


    def valid(self):
        episode_accumulated_rewards = np.empty((self.env_setting["valid_graph_nbr"], self.training_setting["valid_episodes"]))
        for r in range(self.env_setting["valid_graph_nbr"]):
            for episode in range(self.training_setting["valid_episodes"]):
                # in valid set
                g_id = self.env_setting["graph_pool_n"] - self.env_setting["valid_graph_nbr"] + r
                self.environment.init_graph(g_id)
                self.nature.init_graph(g_id)
                self.agent.init_graph(g_id)
                print(f"-"*10 + f" valid : {episode} round in {r} valid graph, graph-{g_id}")

                # sample node feature
                ft_id = 0
                self.environment.init_n_feat(ft_id)
                self.nature.init_n_feat(ft_id)
                self.agent.init_n_feat(ft_id)

                # sample z: random
                hyper_id = 0
                self.environment.init_hyper(hyper_id)
                self.nature.init_hyper(hyper_id)
                self.agent.init_hyper(hyper_id)

                #
                self.environment.reset()
                if self.training_setting["with_nature"]:
                    self.nature.reset()
                    if self.env_setting["hyper_way"] == "rl_nature":
                        nature_state, _ = self.environment.get_seed_state()
                        z_action_pair_lst = self.nature.act(nature_state)
                        z_new = self.environment.step_hyper(z_action_pair_lst)

                # main agent
                self.agent.reset()
                cumul_reward = 0.

                sub_reward = []
                sub_loss = 0

                for i in range(self.environment.budget):
                    # print(f"---------- sub step {i}")
                    state, feasible_action = self.environment.get_seed_state()  # [1, N]
                    action = self.agent.act(state, feasible_action, "valid")
                    print(f"main agent action is {action} ")
                    next_state, reward, done = self.environment.step_seed(i, action)
                    print(f"get reward is {reward}")


                    cumul_reward += reward
                    sub_reward.append(reward)

                # record result
                episode_accumulated_rewards[r, episode] = cumul_reward

        return episode_accumulated_rewards


    def train(self):

        for episode in range(self.training_setting["train_episodes"]):  # one-step, adversary is a bandit
            self.global_iter += 1

            # sample graph
            g_id = random.randint(0, self.env_setting["graph_pool_n"] - self.env_setting["valid_graph_nbr"])
            self.environment.init_graph(g_id)
            self.nature.init_graph(g_id)
            self.agent.init_graph(g_id)
            print(f"-----------this is -- {self.global_iter} iteration,  training in graph {g_id}")

            # sample node feature
            ft_id = 0
            self.environment.init_n_feat(ft_id)
            self.nature.init_n_feat(ft_id)
            self.agent.init_n_feat(ft_id)

            # sample z
            hyper_id = 0
            self.environment.init_hyper(hyper_id)
            self.nature.init_hyper(hyper_id)
            self.agent.init_hyper(hyper_id)

            #
            self.environment.reset()
            if self.training_setting["with_nature"]:
                self.nature.reset()
                nature_state, _ = self.environment.get_seed_state()
                z_action_pair_lst = self.nature.act(nature_state)
                z_new = self.environment.step_hyper(z_action_pair_lst)

            # main agent
            self.agent.reset()
            cumul_reward = 0.

            sub_reward = []
            sub_loss = 0
            for i in range(self.environment.budget):
                # print(f"---------- sub step {i}")
                state, feasible_action = self.environment.get_seed_state()  # [1, N]
                action = self.agent.act(state, feasible_action, "train")
                print(f"main agent action is {action} ")
                next_state, reward, done = self.environment.step_seed(i, action)
                print(f"get reward is {reward}")

                # add to buffer
                if self.main_setting["agent_method"] == 'rl':
                    sample = [state, action, reward, next_state, done, g_id, ft_id, hyper_id]
                    self.agent.remember(sample)
                elif self.main_setting["agent_method"] == 'random':
                    pass

                cumul_reward += reward
                sub_reward.append(reward)

                # get sample and update the main model, GAT
                if self.main_setting["agent_method"] == "rl":
                    loss = self.agent.update(i)
                    self.main_loss.append(loss)
                    sub_loss += loss
                elif self.main_setting["agent_method"] == "random":
                    pass
                # print(f"loss is {loss}")



            # print(f"cumulative reward is {cumul_reward}")
            # plot
            # plt.plot(range(self.environment.budget), sub_reward)
            # plt.title("reward per step")
            # plt.show()

            if self.training_setting["with_nature"]:
                # nature agent
                self.nature.remember(nature_state, z_action_pair_lst, -cumul_reward)
                # get a trajectory and update the nature model
                act_loss_nature, cri_loss_nature = self.nature.update()
            # print(f"actor loss {act_loss_nature} critic loss {cri_loss_nature}")
            # self.nature_critic_loss.append(cri_loss_nature.item())
            # self.nature_actor_loss.append(act_loss_nature.item())

            self.writer.add_scalar(
                f'main/GPU={self.device_setting["use_cuda"]}/nature={self.training_setting["with_nature"]}/cumulative reward per episode',
                cumul_reward, self.global_iter)

            if self.main_setting["agent_method"] == "rl":
                self.writer.add_scalar(
                    f'main/GPU={self.device_setting["use_cuda"]}/nature={self.training_setting["with_nature"]}/mean loss ',
                    sub_loss / self.environment.budget, self.global_iter)
            if self.training_setting["with_nature"]:
                self.writer.add_scalar(f'nature/GPU={self.device_setting["use_cuda"]}/actor loss ', act_loss_nature.item(),
                                  self.global_iter)
                self.writer.add_scalar(f'nature/GPU={self.device_setting["use_cuda"]}/critic loss ', cri_loss_nature.item(),
                                  self.global_iter)

            # validation
            if self.global_iter % self.training_setting["valid_every"] == 0:
                print("-"*10+ f"in checkpoint inter: {self.global_iter}")
                episode_accumulated_rewards = self.valid()      # 5， 20
                print(episode_accumulated_rewards)
                average_accumulated_rewards = np.mean(episode_accumulated_rewards, axis=1)      # 5, 1dims
                for r in range(self.env_setting["valid_graph_nbr"]):
                    g_id = self.env_setting["graph_pool_n"] - self.env_setting["valid_graph_nbr"] + r
                    self.writer.add_scalar(f"valid in graph: {g_id}/ env z by: {self.env_setting['hyper_way']}", average_accumulated_rewards[r], self.global_iter)

