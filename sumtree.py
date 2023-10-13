import numpy as np


class SumTree:
    """
    SumTree

    Modified from: https://github.com/rlcode/per
    ref: https://www.fcodelabs.com/2019/03/18/Sum-Tree-Introduction/

    A binary tree data structure where the parent’s value is the sum of its children
    >>> tree = SumTree(5)
    >>> tree.add(1, 3)
    >>> tree.add(2, 4)
    >>> tree.add(3, 5)
    >>> tree.add(4, 6)
    >>> tree.add(6, 11)
    >>> print(tree.get(4))
    >>> print(tree.get(20))
    >>> print(tree.get(3))
            16 <--- sum of priority
          /    \
       10       6 
      /  \       
     3    7       
    / \  / \  
    1 2  3 4       <--- priority of each value o(log_2 n)

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)        # 按照append顺序存对应的数据
        self.n_entries = 0      # 叶节点，即样本个数
        self.write = 0
    
    # update to the root node
    def _propagate(self, idx, diff):
        parent = (idx - 1) // 2
        self.tree[parent] += diff
        if parent != 0:
            self._propagate(parent, diff)

    # find sample on leaf node
    def _retrieve(self, idx, p):
        left = 2 * idx + 1
        right = left + 1

        #leaf node
        if left >= len(self.tree):
            return idx

        if p <= self.tree[left]:
            return self._retrieve(left, p)
        else:
            return self._retrieve(right, p - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        diff = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, diff)

    # get priority and sample
    def get(self, p):
        # return idx, priority, data. 
        # idx是在存树的数据结构中的索引， 
        idx = self._retrieve(0, p)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])