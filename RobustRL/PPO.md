# PPO

- actor: 输入一个state，输出所有action的概率分布

## update(transition)

(根据PPO原代码)

1. 获取states, actions，rewards，next_states
2. 用critic(next_states)计算td_target，critic(states)， 计算td_delta



## on policy

用一个episode内的transition，用积累的所有的去更新，不用随机采样；每个episode重新记录



## off policy

用一个buffer 记录下所有的transition，从buffer中随机采样