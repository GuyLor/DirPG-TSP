
import torch
import numpy as np
import copy
import kruskals_cpp
import dirpg_cpp
from utils import utils_gumbel


class CPP_manager:
    def __init__(self, root_node, batch_size, graph_size):


        root_cpp_converter = dirpg_cpp.convert_python_to_cpp(*root_node.get())

        self.batched_heaps = dirpg_cpp.BatchedHeaps(root_cpp_converter, batch_size, graph_size)
        self.objects_dict = {root_cpp_converter: batch_size}
        self.batch_size = batch_size

    def pop_batch(self):

        (parents, to_remove), trajs = self.batched_heaps.pop_batch()
        for converter in to_remove:
            self.objects_dict[converter] -= 1
            if self.objects_dict[converter] == 0:
                del self.objects_dict[converter]

        return Node.create_node_from_cpp(parents), trajs

    def push_batch(self, node, to_avoid):
        cpp_converter = dirpg_cpp.convert_python_to_cpp(*node.get())
        self.objects_dict.update({cpp_converter: copy.copy(self.batch_size)})
        self.batched_heaps.push_batch(cpp_converter, to_avoid)


class Trajectory:
    main_queue = None

    def __init__(self):

        self.num_candidates = 0
        self.t_opt = None
        self.t_direct = None

    def set_trajectory(self, cpp_traj):
        self.num_candidates += 1
        if self.num_candidates == 1:  # if t_opt
            self.t_opt = cpp_traj
            self.t_direct = cpp_traj
            self.main_queue.lower_bound[cpp_traj.idx] = cpp_traj.objective

        else:
            if cpp_traj.objective > self.t_direct.objective:
                self.t_direct = cpp_traj
                self.main_queue.lower_bound[cpp_traj.idx] = cpp_traj.objective

    def print_trajectory(self):
        print('--------  Trajectory  ---------')
        print('t_opt:  ', self.t_opt)
        print('t_direct:  ', self.t_direct)
        print('t_direct objective:  ', self.t_direct.objective)
        print('t_opt objective:  ', self.t_opt.objective)
        print('-------------------------------')


class Node:
    epsilon = 1.0
    alpha = 2.0  # 2: full walk on the MST
    dynamic_weighting = True
    graph_size = 20
    heuristic = ''

    mst_edges = None
    mst_val = None

    def __init__(self,
                 ids,  # ids of sample in the batch - maybe to remove
                 t,  # time-step (not necessarily the same t for each sample)
                 next_actions,  # mask: 1 available, 0 unavailable
                 not_visited,  # mask: 1 available, 0 unavailable (note that next_actions!=not_visited)
                 prefix,  # e.g.: [2,6,1,0,-1,-1,-1,-1]
                 lengths,
                 cur_coord,
                 done,
                 logprob_so_far,
                 bound_togo,
                 max_gumbel,
                 is_t_opt):  # true: opt, false: direct

        self.ids = ids
        self.t = t
        self.next_actions = next_actions
        self.not_visited = not_visited
        self.prefix = prefix

        self.lengths = lengths
        self.cur_coord = cur_coord
        self.done = done
        self.logprob_so_far = logprob_so_far
        self.max_gumbel = max_gumbel
        self.is_t_opt = is_t_opt

        if self.dynamic_weighting:
            self.alpha = 1 + self.alpha * (1 - self.t / self.graph_size)

        self.bound_togo = bound_togo

        self.eps_reward = self.epsilon * (self.lengths + self.alpha * self.bound_togo)

        self.priority = self.get_priority()
        self.objective = self.priority
        self.upper_bound = self.priority  # self.get_upper_bound()

    def get_priority(self, alpha=2):
        return self.max_gumbel + self.eps_reward

    def get_priority_max_gumbel(self):
        return self.max_gumbel

    def get_upper_bound(self, city=None):
        return self.max_gumbel + self.epsilon * (self.lengths + 1.0 * self.bound_togo)  # self.lengths + self.bound_length_togo()

    def get_objective(self):
        """Computes the objective of the trajectory.
        Only used if a node is terminal.
        """
        return self.max_gumbel + self.epsilon * self.lengths

    def get(self):
        return self.priority, self.is_t_opt, self.done, self.ids, self.t, self.next_actions, self.not_visited,\
               self.prefix, self.lengths, self.cur_coord, self.logprob_so_far, self.bound_togo, self.max_gumbel,\
               self

    def pack(self):
        """
        pack node to cpp API: float, bool, obj_adrs, int (row in the original batch)
        """
        return [[self.priority[i].item(),
                 self.is_t_opt[i].item(),
                 self.done[i].item(),
                 self,
                 i] for i in range(self.ids.size(0))]

    def print(self):
        print(' -----------    Node     -----------')
        print('ids:  ', self.ids)
        # print('first_a:  ', self.first_a)
        print('prefix:  ', self.prefix)
        print('not visited:  ', self.not_visited)
        print('next_actions:  ', self.next_actions)
        print('t:  ', self.t)
        # print('distance matrix:')
        print('max_gumbel:  ', self.max_gumbel)
        print('epsilon:   ', self.epsilon)
        print('alpha:   ', self.alpha)
        print('alpha * len_togo:   ', self.alpha * self.bound_togo)
        print('eps_reward: ', self.eps_reward, self.eps_reward.size())
        print('priority: ', self.priority)
        print('objective: ', self.objective)
        # print('upper bound: ',self.upper_bound)
        print('lengths:  ', self.lengths)
        print('bound length togo: ', self.bound_togo)
        print('done:  ', self.done)
        print('logprob_so_far:  ', self.logprob_so_far)

        print('is_t_opt:  ', self.is_t_opt)
        print(' -------------------------------')

    @staticmethod
    def create_node_from_cpp(args):
        return Node(ids=args[0],
                    t=args[1],
                    next_actions=args[2].bool(),
                    not_visited=args[3].bool(),
                    prefix=args[4],
                    lengths=args[5],
                    cur_coord=args[6],
                    done=args[7].bool(),
                    logprob_so_far=args[8],
                    bound_togo=args[9],
                    max_gumbel=args[10],
                    is_t_opt=args[11].bool())


class PriorityQueue:
    def __init__(self,
                 init_state,
                 graphs,
                 epsilon,
                 search_params,
                 inference=False
                 ):

        special_action = init_state.prev_a
        not_visited = ~init_state.visited.bool()

        dummy_idx = special_action.repeat(1, init_state.loc.size(-1)).unsqueeze(1)
        self.first_coord = torch.gather(init_state.loc, 1, dummy_idx)

        self.graph_size = graphs.size(1)
        self.batch_size = graphs.size(0)
        ##########################################
        #         global nodes parameters        #
        Node.alpha = search_params.alpha
        Node.epsilon = epsilon
        Node.dynamic_weighting = search_params.dynamic_weighting
        Node.heuristic = search_params.heuristic
        Node.graph_size = self.graph_size
        ##########################################
        #      TODO: sort edges, kruskals_cpp

        # self.mst_edges = mst.prim_pytorch(distance_mat)
        # self.mst_val = self.mst_edges.sum()

        device = not_visited.device
        prefix = -torch.ones(self.batch_size, self.graph_size, device=device, dtype=torch.long)
        # prefix = p.scatter_(-1,torch.zeros(self.batch_size, dtype=torch.long), special_action) #
        prefix[:, 0] = special_action.squeeze(-1)
        root_node = Node(ids=init_state.ids,
                         t=init_state.i,
                         next_actions=not_visited,  # torch.tensor(not_visited),  # number of cities
                         not_visited=not_visited,
                         prefix=prefix,
                         lengths=torch.zeros(self.batch_size, 1, device=device),
                         cur_coord=self.first_coord,
                         done=torch.zeros(self.batch_size, 1, device=device, dtype=torch.bool),
                         logprob_so_far=torch.zeros(self.batch_size, 1, device=device),
                         bound_togo=torch.ones(self.batch_size, 1, device=device),  # self.mst_val,
                         max_gumbel=utils_gumbel.sample_gumbel(self.batch_size).unsqueeze(-1),
                         is_t_opt=torch.ones(self.batch_size, device=device, dtype=torch.bool))

        self.cpp_heaps = CPP_manager(root_node, self.batch_size, self.graph_size)
        Trajectory.main_queue = self
        self.batch_trajectories = [Trajectory() for _ in range(self.batch_size)]
        self.lower_bound = [-float('Inf') for _ in range(self.batch_size)]

        self.t_opt = None
        self.t_direct = None

        self.num_interactions = 0
        self.first_improvement = search_params.first_improvement
        self.max_interactions = search_params.max_interactions

        self.dynamic_weighting = search_params.dynamic_weighting
        self.inference = inference

        self.p = torch.ones(self.batch_size, dtype=torch.bool) * (not search_params.not_prune)
        self.prune = torch.zeros(self.p.size(), dtype=torch.bool)

    def pop(self):
        current_node, trajs_list = self.cpp_heaps.pop_batch()
        # print('parents_list: ', len(parents_list), 'trajs_list: ', len(trajs_list))
        for t in trajs_list:
            self.batch_trajectories[t.idx].set_trajectory(t)

        """
        if self.num_interactions >= self.max_interactions:
            return 'break'

        if self.prune and self.lower_bound > parent.upper_bound:
            self.prune_count += 1
            return self.pop()
        """
        return current_node

    def expand(self, state, current_node, logprobs):
        self.num_interactions += 1
        special_action = state.prev_a
        # visited = state.visited_.bool()
        not_visited = current_node.not_visited.scatter(-1, special_action.unsqueeze(-1), False)
        is_done = torch.all(~not_visited.transpose(1, 2), dim=1, keepdim=False)
        prefix = current_node.prefix.scatter(-1, state.i - 1, special_action)
        cur_coord = state.loc[current_node.ids, special_action]

        lengths = -state.lengths  # .squeeze(-1)

        lengths = torch.where(is_done,
                              lengths - (self.first_coord - cur_coord).norm(p=2, dim=-1),
                              lengths)  # add distance to first node if it's complete trajectory

        lp_special = torch.gather(logprobs, -1, special_action)

        special_child = Node(
            ids=state.ids,
            t=current_node.t + 1,
            not_visited=not_visited,
            prefix=prefix,
            lengths=lengths,
            cur_coord=cur_coord,
            done=is_done,
            logprob_so_far=current_node.logprob_so_far + lp_special,
            max_gumbel=current_node.max_gumbel,
            next_actions=not_visited,
            bound_togo=current_node.bound_togo,
            is_t_opt=current_node.is_t_opt)

        # to_prune = self.prune | (special_child.upper_bound < self.lower_bound)
        to_prune = self.prune

        self.cpp_heaps.push_batch(special_child, to_prune.squeeze(0))
        """
        if self.prune and special_child.upper_bound < self.lower_bound:
            self.prune_count += 1
        else:
            heapq.heappush(self.queue, special_child)

        """
        # Sample the max gumbel for the non-chosen actions and create an "other
        # children" node if there are any alternatives left.

        other_actions = current_node.next_actions.scatter(-1, special_action.unsqueeze(-1), False)

        assert torch.all(other_actions.sum(-1) == current_node.next_actions.sum(
            -1) - 1), 'other_actions.sum(-1): {} current_node.next_actions.sum(-1) - 1): {}'.format(
            other_actions.sum(-1), current_node.next_actions.sum(-1) - 1)

        if not self.inference:
            exp = torch.exp(logprobs)

            exp[~other_actions.squeeze(1)] = 0
            other_max_location = torch.log(exp.sum(-1)).unsqueeze(-1)

            other_max_gumbel = utils_gumbel.sample_truncated_gumbel(current_node.logprob_so_far + other_max_location,
                                                                    current_node.max_gumbel)

            ignore = (other_max_location == -np.inf).squeeze(-1)

            other_children = Node(
                ids=current_node.ids,
                t=current_node.t,
                not_visited=current_node.not_visited,
                prefix=current_node.prefix,
                lengths=current_node.lengths,
                cur_coord=current_node.cur_coord,
                done=current_node.done,
                logprob_so_far=current_node.logprob_so_far,
                max_gumbel=other_max_gumbel,
                next_actions=other_actions,
                bound_togo=current_node.bound_togo,  # + self.mst_edges[special_action].sum(),
                is_t_opt=current_node.is_t_opt * False)

            """
            if self.prune and other_children.upper_bound < self.lower_bound:
                self.prune_count += 1
            else:
                heapq.heappush(self.queue, other_children)
            """

            # to_prune = self.prune | (other_children.upper_bound < self.lower_bound) | ignore
            to_prune = (self.prune | ignore)
            self.cpp_heaps.push_batch(other_children, to_prune)


