import torch
import torch.nn.functional as F
import utils


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        '''
        如果动作离散：输入一个state， 输出action分布概率，[0.1, 0.3, ...]
        这里动作连续：输入一个state，输出每个action的分布均值和方差
        :param state_dim:
        :param hidden_dim:
        :param action_dim:
        '''
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        # return F.softmax(self.fc2(x), dim=1)
        mu = 2.0 * torch.tanh(self.fc_mu(x))       # 映射到 【-2，2】
        std = F.softplus(self.fc_std(x))
        return mu, std

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        '''
        输入一个state，输出该state的scalar value，
        :param state_dim:
        :param hidden_dim:
        '''
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPOContinuousAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, lmbda, eps, epochs):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim)
        self.critic = ValueNet(state_dim, hidden_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps      # 截断
        self.epochs = epochs

    def act(self, state):
        state = torch.tensor([state], dtype=torch.float)
        # probs = self.actor(state)
        mu, sigma= self.actor(state)
        # action_dist = torch.distributions.Categorical(probs)       # 根据actor给出的各个action的概率得到的所有action的分布
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()       # distribution + sample 来采样, mu sigma是向量，采样得到的action也是一样大小的向量
        # return action.item()
        return [action.item()]

    def update(self, transition):
        # 一个transition（虽然这里变量是states）， 一个list: [state, action, next_state, reward]
        states = torch.tensor(transition[0], dtype=torch.float)
        actions = torch.tensor(transition[1])
        rewards = torch.tensor(transition[2], dtype=torch.float)
        rewards = (rewards + 8.0) / 8.0     # ？ 和TRPO一样,对奖励进行修改,方便训练
        next_states = torch.tensor(transition[3], dtype=torch.float)

        # 时序差分target
        td_target = rewards + self.gamma * self.critic(next_states)     # ? dones是什么
        td_delta = td_target - self.critic(states)
        advantage = utils.compute_advantage(self.gamma, self.lmbda, td_delta)
        # old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())        # 后面的训练中，梯度更新并不改变actor ???
        # 动作时正态分布值 0-1
        old_log_probs = action_dists.log_prob(actions)     # 该分布下的概率密度函数，输入值得到概率值



        # 每次更新都对网络迭代很多次， 每次的new policy更新，old policy不更新
        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

