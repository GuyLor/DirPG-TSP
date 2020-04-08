import numpy as np
import copy
import time
import torch
from torch.nn import DataParallel
import heapq
from utils import utils_gumbel
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from problems.tsp.state_tsp import StateTSP


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
        self.alpha = 2.0  # 2: full walk on the MST

        self.cur_coord = cur_coord

        self.done = done
        self.logprob_so_far = logprob_so_far
        self.max_gumbel = max_gumbel
        self.next_actions = next_actions

        self.t_opt = t_opt  # true: opt, false: direct
        self.dfs_like = dfs_like
        self.upper_bound = self.get_upper_bound()
        self.priority = self.get_priority()
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

    def get_priority(self):
        return self.max_gumbel + self.epsilon * self.get_upper_bound()

    def get_priority_max_gumbel(self):
        return self.max_gumbel

    def get_upper_bound(self):
        return self.lengths + self.bound_length_togo()

    def bound_length_togo(self):
        return -self.alpha * minimum_spanning_tree(self.dist).toarray().sum()

    def get_objective(self):
        """Computes the objective of the trajectory.
        Only used if a node is terminal.
        """
        return self.lengths  #  self.max_gumbel + self.epsilon * self.lengths

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
                 max_interactions=200,
                 max_pop=10000,
                 dfs_like=False,
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
        self.keep_searching = True
        self.start_time = float('Inf')
        # self.max_search_time = max_search_time
        self.num_interactions = 0
        self.max_interactions = max_interactions
        self.max_pop = max_pop
        self.dfs_like = dfs_like
        self.inference = inference
        self.prune = False

        self.lower_bound = -float('Inf')

    def pop(self):
        if not self.queue:
            return 'break'

        parent = heapq.heappop(self.queue)
        self.current_node = parent

        if self.num_interactions >= self.max_interactions:
            #print('prune_count: ', self.prune_count)
            #print('trajectories_list: ', len(self.trajectories_list))
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
            self.lower_bound = t.objective
            if self.inference:
                return 'break'
        else:
            if t.objective > self.t_direct.objective:
                self.t_direct = t
                self.lower_bound = t.objective
                if not self.keep_searching:
                    print('*****  priority(direct) > priority(opt)   *****')
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
        idx = torch.ones(self.current_node.dist.size(-1),
                         dtype=torch.long,
                         device=state.loc.device)*special_action
        dist = self.current_node.dist.scatter(1, idx.unsqueeze(-1), 0).scatter_(0, idx.unsqueeze(0), 0)
        special_child = Node(
            id=self.current_node.id,
            first_a=self.current_node.first_a,
            not_visited=not_visited,
            prefix=self.current_node.prefix + [special_action],
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


class DirPG:
    def __init__(self,
                 model,
                 ):
        model = model.module if isinstance(model, DataParallel) else model
        self.encoder = model
        self.decoder = model.decoder

        self.interactions = 0

    def train_dirpg(self, batch, epsilon=1.0):
        embeddings = self.encoder(batch, only_encoder=True)
        state = self.encoder.problem.make_state(batch)
        fixed = self.encoder.precompute(embeddings)
        Node.epsilon = epsilon
        opt_direct, interactions = self.sample_t_opt_search_t_direct(state, fixed, inference=False)
        self.interactions += interactions
        opt, direct = zip(*opt_direct)
        opt_actions, opt_objectives = self.stack_trajectories_to_batch(opt, device=batch.device)
        direct_actions, direct_objectives = self.stack_trajectories_to_batch(direct, device=batch.device)

        log_p_opt, opt_length = self.run_actions(state, opt_actions, batch, fixed)
        log_p_direct, direct_length = self.run_actions(state, direct_actions, batch, fixed)

        direct_loss = (log_p_opt - log_p_direct)/(epsilon+1e-7)

        return direct_loss, {'opt_cost': opt_length,
                             'direct_cost': direct_length,
                             'opt_objective': opt_objectives,
                             'direct_objective': direct_objectives,
                             'interactions': self.interactions}

    def sample_t_opt_search_t_direct(self, state, fixed, inference=False):
        start_encoder = time.time()

        batch_size = state.ids.size(0)
        _, state = self.forward_and_update(state, fixed)
        queues = [PriorityQueue(state[i],
                                state.dist[idx],
                                inference=inference) for idx, i in enumerate(torch.tensor(range(batch_size)))]

        batch_t, interactions = [], []
        pop_t, model_t, stack_t, expand_t = [], [], [], []

        end_beg = time.time()
        inner_s, inner_o = 0, 0
        while queues:  # batch
            start = time.time()
            nodes = []
            copy_queues = copy.copy(queues)
            for queue in copy_queues:
                parent = queue.pop()

                if parent == 'break':
                    if False and queue.id == 0:
                        print('prune_count: ', queue.prune_count)
                        print('trajectories_list: ', len(queue.trajectories_list))
                    interactions.append(queue.num_interactions)
                    batch_t.append((queue.t_opt, queue.t_direct))
                    queues.remove(queue)
                    continue
                else:
                    nodes.append(parent)
            after_pop = time.time()
            if len(nodes) > 0:

                batch_state = state.stack_state(nodes)
                after_stack = time.time()
                log_p, state = self.forward_and_update(batch_state, fixed)

                # log_p = log_p.numpy()
                after_model = time.time()

                idx = torch.tensor(range(len(queues)))
                for i, queue in zip(idx, queues):
                    a, b = queue.expand(state[i], log_p[i])
                    inner_s += a
                    inner_o += b

                after_expand = time.time()
                pop_t.append(after_pop - start)
                stack_t.append(after_stack - after_pop)
                model_t.append(after_model - after_stack)
                expand_t.append(after_expand - after_model)

        t = end_beg - start_encoder
        """
        print('---------- our time detailed  -------')
        print('encoder: ', t)
        print('pop: ', np.sum(pop_t))
        print('stack: ', np.sum(stack_t))
        print('model: ', np.sum(model_t))
        print('expand: ', np.sum(expand_t))
        print('expand special: ', inner_s)
        print('expand other: ', inner_o)
        print('expand oh: ', np.sum(expand_t) - (inner_s+inner_o))
        print('total: ', t + np.sum(pop_t) + np.sum(stack_t) + np.sum(model_t) + np.sum(expand_t))
        """
        return batch_t, np.mean(interactions)

    def forward_and_update(self, batch, fixed):
        with torch.no_grad():
            self.decoder.eval()

            log_p, _ = self.decoder(fixed[:batch.ids.size(0)], batch)

        log_p = log_p[:, 0, :]

        _, selected = utils_gumbel.sample_gumbel_argmax(log_p)
        #selected = torch.argmax(log_p, -1)
        state = batch.update(selected, update_length=False)

        return log_p, state

    def stack_trajectories_to_batch(self, trajectories, device):
        actions, objectives = [], []
        for t in trajectories:
            actions.append(t.actions)
            objectives.append(t.objective)

        return torch.tensor(actions, device=device).split(1, 1), np.mean(objectives)

    def stack_lengths_to_batch(self, trajectories, device):
        return torch.tensor([t.length for t in trajectories],  device=device)

    def run_actions(self, state, actions, batch, fixed):
        outputs = []

        for action in actions:
            log_p, mask = self.decoder(fixed, state) #self._get_log_p(fixed, state)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            log_p = log_p.squeeze(1)

            state = state.update(action.squeeze(1), update_length=True)
            # Collect output of step
            log_p = log_p.gather(1, action).squeeze(-1)
            outputs.append(log_p)

        # Collected lists, return Tensor
        a = torch.cat(actions,1)
        d = batch.gather(1, a.unsqueeze(-1).expand_as(batch))

        lengths = state.lengths + (d[:, 0] - d[:, -1]).norm(p=2, dim=1).unsqueeze(-1)
        outputs = torch.stack(outputs,1)

        return outputs, lengths
