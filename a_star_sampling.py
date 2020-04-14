
import time
import torch
import numpy as np
import heapq
from utils import utils_gumbel
from scipy.sparse.csgraph import minimum_spanning_tree
import prim

class Trajectory:
    def __init__(self, actions, gumbel, length, objective):
        self.actions = actions
        self.gumbel = gumbel
        self.length = length
        self.objective = objective

    def print_trajectory(self):
        print('--------  Trajectory  ---------')
        print('actions:  ', self.actions)
        print('gumbel:  ', self.gumbel)
        print('length:  ', self.length)
        print('objective:  ', self.objective)
        print('-------------------------------')


class Node:
    epsilon = 1.0

    def __init__(self,
                 id,
                 first_a,
                 next_actions,
                 not_visited,
                 prefix=[],
                 lengths=0.0,
                 cur_coord=None,
                 done=False,
                 logprob_so_far=0,
                 dist=None,
                 max_gumbel=None,
                 t_opt=True,
                 dfs_like=True):  # How much total objective t_opt achieved.

        if max_gumbel is None:
            max_gumbel = utils_gumbel.sample_gumbel(0)

        self.id = id
        self.first_a = first_a

        self.not_visited = not_visited
        self.prefix = prefix
        self.t = len(self.prefix)

        self.dist = dist
        self.lengths = lengths
        self.alpha = 1.0  # 2: full walk on the MST

        self.cur_coord = cur_coord

        self.done = done
        self.logprob_so_far = logprob_so_far
        self.max_gumbel = max_gumbel
        self.next_actions = next_actions

        self.t_opt = t_opt  # true: opt, false: direct
        self.dfs_like = dfs_like
        self.upper_bound = self.get_upper_bound(1.0)
        self.priority = self.get_priority(2.0)
        self.objective = self.get_objective()


    def __lt__(self, other):
        if self.t_opt == other.t_opt and not self.dfs_like: #false==false
            return self.priority > other.priority
        elif self.t_opt or self.dfs_like:
            """
            This is how we sample t_opt, the starting node is with t_opt=True and
            its 'special child' will be also with t_opt=True and because we always returning true
            when we compare it to the 'other childrens' (where t_opt=False) the special path is sampled (speical childs only)
            """
            return True

    def get_priority(self, alpha=2):
        return self.max_gumbel + self.epsilon * self.get_upper_bound(alpha)

    def get_priority_max_gumbel(self):
        return self.max_gumbel

    def get_upper_bound(self, alpha=1):
        return self.lengths + self.bound_length_togo(alpha)

    def bound_length_togo(self, alpha):
        return -alpha * prim.mst(self.dist.numpy()) if len(self.dist) != 0 else 0

    def get_objective(self):
        """Computes the objective of the trajectory.
        Only used if a node is terminal.
        """
        return self.max_gumbel + self.epsilon * self.lengths

    def print(self):
        print(' -----------    Node     -----------')
        print('id:  ', self.id)
        print('first_a:  ', self.first_a)
        print('prefix:  ', self.prefix)
        print('not visited:  ', self.not_visited)
        print('next_actions:  ', self.next_actions)
        print('t:  ', self.t)
        print('distance matrix:')
        print(self.dist)
        print('upper bound: ',self.upper_bound)
        print('lengths:  ', self.lengths)
        print('bound length togo: ', self.bound_length_togo())
        print('done:  ', self.done)
        print('logprob_so_far:  ', self.logprob_so_far)
        print('max_gumbel:  ', self.max_gumbel)
        print('t_opt:  ', self.t_opt)
        print(' -------------------------------')


class PriorityQueue:
    def __init__(self,
                 init_state,
                 distance_mat,
                 inference=False,
                 prune=False,
                 max_interactions=200,
                 first_improvement=False,
                 dfs_like=True,
                 ):
        self.queue = []

        init_state = init_state._replace(first_a=init_state.first_a.squeeze(0),
                                         prev_a=init_state.prev_a.squeeze(0),
                                         visited_=init_state.visited_.squeeze(0),
                                         lengths=init_state.lengths.squeeze(0),
                                         cur_coord=init_state.cur_coord.squeeze(0),
                                         ids=init_state.ids.squeeze(0),
                                         i=init_state.i.squeeze(0))

        special_action = init_state.prev_a.item()
        not_visited = [i for i in range(init_state.loc.size(1)) if i != special_action]
        self.first_coord = init_state.loc[init_state.ids, special_action]

        root_node = Node(id=init_state.ids,
                         first_a=init_state.first_a.item(),
                         next_actions=not_visited, # torch.tensor(not_visited),  # number of cities
                         not_visited=not_visited,
                         prefix=[special_action],
                         dist=distance_mat,
                         lengths=0.0,
                         cur_coord=self.first_coord,
                         max_gumbel=utils_gumbel.sample_gumbel(0),
                         t_opt=True)

        heapq.heappush(self.queue, root_node)
        self.current_node = root_node
        self.id = init_state.ids.item()

        self.trajectories_list = []
        self.t_opt = None
        self.t_direct = None

        self.prune_count = 0

        self.start_search_direct = False
        self.first_improvement = first_improvement
        self.start_time = float('Inf')
        # self.max_search_time = max_search_time
        self.num_interactions = 0
        self.max_interactions = max_interactions
        self.dfs_like = dfs_like
        self.inference = inference
        self.prune = prune

        self.lower_bound = -float('Inf')

    def pop(self):
        if not self.queue:
            return 'break'

        parent = heapq.heappop(self.queue)
        self.current_node = parent

        if self.num_interactions >= self.max_interactions:
            return 'break'

        if self.prune and self.lower_bound > parent.upper_bound:
            self.prune_count += 1
            return self.pop()

        # Start the search time count
        if not parent.t_opt and not self.start_search_direct:
            self.start_time = time.time()
            self.start_search_direct = True

        if parent.done:
            return self.set_trajectory(parent)

        return parent

    def set_trajectory(self, node):

        t = Trajectory(actions=node.prefix,
                       gumbel=node.max_gumbel,
                       length=node.lengths - (self.first_coord - node.cur_coord).norm(p=2, dim=-1),
                       objective=node.objective)

        self.trajectories_list.append(t)

        if node.t_opt:
            self.t_opt = t
            self.t_direct = t
            self.lower_bound = t.length
            if self.inference:
                return 'break'
        else:
            if t.objective > self.t_direct.objective:
                self.t_direct = t
                self.lower_bound = t.length
                if self.first_improvement:
                    #print('*****  priority(direct) > priority(opt)   *****')
                    return 'break'

        if self.queue:
            return self.pop()
        else:
            # print('break')
            return 'break'

    def expand(self, state, logprobs):
        self.num_interactions += 1
        special_action = state.prev_a.item()
        s = time.time()
        not_visited = [i for i in self.current_node.not_visited if i != special_action]
        cur_coord = state.loc[self.current_node.id, special_action]
        length = -(cur_coord - self.current_node.cur_coord).norm(p=2, dim=-1)

        updated_prefix = self.current_node.prefix + [special_action]

        dist = np.delete(np.delete(self.current_node.dist, updated_prefix, 0), updated_prefix, 1)
        special_child = Node(
            id=self.current_node.id,
            first_a=self.current_node.first_a,
            not_visited=not_visited,
            prefix=updated_prefix,
            lengths=self.current_node.lengths + length,
            cur_coord=cur_coord,
            done=len(not_visited) == 0,
            logprob_so_far=self.current_node.logprob_so_far + logprobs[special_action],
            max_gumbel=self.current_node.max_gumbel,
            next_actions=not_visited,
            dist=dist,
            t_opt=self.current_node.t_opt,
            dfs_like=self.dfs_like)

        if self.prune and special_child.upper_bound < self.lower_bound:
            self.prune_count += 1

        else:
            heapq.heappush(self.queue, special_child)

        # Sample the max gumbel for the non-chosen actions and create an "other
        # children" node if there are any alternatives left.

        m = time.time()
        other_actions = [i for i in self.current_node.next_actions if i != special_action]

        assert len(other_actions) == len(self.current_node.next_actions) - 1
        if other_actions and not self.inference:
            other_max_location = utils_gumbel.logsumexp(logprobs[other_actions])
            other_max_gumbel = utils_gumbel.sample_truncated_gumbel(self.current_node.logprob_so_far + other_max_location,
                                                                    self.current_node.max_gumbel).item()
            other_children = Node(
                id=self.current_node.id,
                first_a=self.current_node.first_a,
                not_visited=self.current_node.not_visited,
                prefix=self.current_node.prefix,
                lengths=self.current_node.lengths,
                cur_coord=self.current_node.cur_coord,
                done=self.current_node.done,
                logprob_so_far=self.current_node.logprob_so_far,
                max_gumbel=other_max_gumbel,
                next_actions=other_actions,
                dist=self.current_node.dist,
                t_opt=False,
                dfs_like=False)

            if self.prune and other_children.upper_bound < self.lower_bound:
                self.prune_count += 1
            else:
                heapq.heappush(self.queue, other_children)

        f = time.time()
        sp = m - s
        oth = f - m
        return sp, oth
