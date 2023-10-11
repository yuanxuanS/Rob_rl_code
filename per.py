

class PER:
    def __init__(self, capacity=1000000, e=0.01, a=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.capacity = capacity
        self.__init_tree()
        self.e = 0.01       # ?
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
    
    def __init_tree(self):
        self.tree = SumTree(self.capacity)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def append(self, sample, error=0):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size=32, global_step=0): 
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            # 采样区间
            a = segment * i
            b = segment * (i + 1)

            #mitigate the data=0 bug
            if i == batch_size-1:
                b -= 0.01
            sampled_priority = random.uniform(a, b)
            #print('[INFO] sampled priority: ', sampled_priority)
            (idx, priority, data) = self.tree.get(sampled_priority)
            #print('[INFO] retrieved priority: ', priority)
            
            #make up for the data=0 bug with resampling
            if type(data) == int:
                sampled_priority = random.uniform(0, a)
                (idx, priority, data) = self.tree.get(sampled_priority)

            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        #print('[INFO] all priorities', self.tree.tree)
        sampling_probabilities = priorities / self.tree.total()     # 把这些优先级归一化
        #print('[INFO] sampling_priorities: ', sampling_probabilities)
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        #print('[INFO] is_weight: ', is_weight)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

    def _last_transit(self):
        return self.tree.data[self.tree.n_entries-1]

    def _temp_update_idxs(self, idxs):
        idxs.append(self.tree.n_entries)
        return idxs