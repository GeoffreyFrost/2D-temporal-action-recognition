"""
Implementation by https://github.com/yysijie/st-gcn / https://arxiv.org/abs/1801.07455
"""

import numpy as np

class Graph():
    """
    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in this paper (https://arxiv.org/abs/1801.07455).
        layout (string): must be one of the follow candidates
        - blazepose: 2D 33 point pose estimation (https://arxiv.org/abs/2006.10204)
        - blazepose_mini: 2D 21 point pose estimaition (smplification of blazepose)
        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """

    def __init__(self,
                 layout='blazepose_mini',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):

        if layout == 'blazepose_mini':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5,6),
                             (0, 7), (7, 8), (8, 9), (9, 10), (7, 11), (11, 12),
                             (12, 13), (7, 14), (14, 15), (14, 18), (15, 16),
                             (16, 17), (18, 19), (19, 20)]

            self.edge = self_link + neighbor_link
            self.center = 7

        elif layout == 'blazepose':
            self.num_node = 33
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(10, 9), (0, 4), (4, 5), (5, 6), (6, 8), (0,1),
                             (1, 2), (2, 3), (3, 7), (12, 11), (12, 14), (14, 16),
                             (16, 22), (16, 18), (18, 20), (20, 16), (11, 13),
                             (13, 15), (15, 21), (15, 17), (17, 19), (19, 15),
                             (12, 24), (11, 23), (24, 23), (24, 26), (26, 28),
                             (28, 32), (32, 30), (30, 28), (23, 25), (25, 27),
                             (27, 29), (29, 31), (31, 27)
                             ]
            self.edge = self_link + neighbor_link
            self.center = 7

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("This strategy does not exist")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # Compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
