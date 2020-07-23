import numpy as np
import copy
import torch
from torch.nn import DataParallel
from dirpg_tsp import a_star_sampling
from utils import utils_gumbel
import dirpg_cpp

def print_cpp_state(state_cpp):
    print("=============  cpp state ============ ")
    print("prev_city: ",state_cpp.batch_prev_city)

    print("t: ", state_cpp.batch_t)
    print("done: ", state_cpp.done)
    print("next_actions: ")
    print(state_cpp.batch_next_actions)
    print("==================================== ")

def print_trajectory(t_opt,t_direct,t):
    print("------- trajectory ----------")
    print(t_opt.actions)
    print(t_direct.actions)
    print(t.num_candidates)
    print("------- ----- ----------")

class DirPG:
    def __init__(self,
                 model,
                 search_params
                 ):
        model = model.module if isinstance(model, DataParallel) else model
        self.encoder = model
        self.decoder = model.decoder
        self.interactions = 0
        self.search_params = search_params

        self.a_star_cpp = dirpg_cpp.AstarSampling(search_params.batch_size,
                                                  search_params.max_interactions,
                                                  search_params.graph_size,
                                                  search_params.epsilon,
                                                  search_params.alpha,
                                                  626)

    def train_dirpg(self, batch, epsilon=1.0):
        to_log = dict.fromkeys(['interactions', 'candidates', 'prune_count', 'bfs', 'dfs', 'jumps']) # TODO cpp output
        self.num_of_non_empty_heaps = batch.size(0)
        self.a_star_cpp.clear()
        embeddings = self.encoder(batch, only_encoder=True)
        fixed = self.encoder.precompute(embeddings)
        init_state = self.encoder.problem.make_state(batch)
        empty_heaps = torch.tensor([True for _ in range(batch.size(0))])
        with torch.no_grad():

            py_state = self.init_batched_priority_queue(batch, init_state, fixed)
            first_coord = py_state.loc[py_state.ids, py_state.first_a, :]
            step = 1
            while step < self.search_params.max_interactions and self.num_of_non_empty_heaps >0:  ## interactions budget
                #print("^^^^^^^^^^^^   ", step, "   ^^^^^^^^^^^^^^^")
                cpp_state = self.a_star_cpp.popBatch()
                if step == 1:
                    # ignore the first pop
                    self.a_star_cpp.non_empty_heaps = empty_heaps

                py_state = self.update_py_state(py_state, cpp_state)
                fixed, py_state = self.filter_empty_heaps(fixed, py_state)

                log_p, next_city = self.decode_and_sample(fixed, py_state)

                cost = self.compute_cost(py_state, next_city, cpp_state.batch_t, first_coord)
                # self.print_trajectory()
                self.a_star_cpp.expand(next_city, log_p, cost)
                step+=1


        self.interactions += self.search_params.max_interactions

        t = self.a_star_cpp.getTrajectories()
        t_opt, t_direct = t.get_t_opt_direct()

        # forward again this time gradients tracking for the backward pass

        log_p_opt = self.run_actions(init_state, t_opt.actions, batch, fixed)  #, opt_length
        log_p_direct = self.run_actions(init_state, t_direct.actions, batch, fixed)  #, direct_length

        direct_loss = (log_p_opt - log_p_direct) / epsilon

        to_log.update({'opt_cost': t_opt.costs,
                       'direct_cost': t_direct.costs,
                       'opt_objective': t_direct.objectives,
                       'direct_objective': t_direct.objectives,
                       'candidates': np.mean(t.num_candidates),
                       'interactions': self.interactions})

        return direct_loss, to_log

    def decode_and_sample(self, fixed, batch_state, first_action=None):

        log_p, _ = self.decoder(fixed, batch_state)
        log_p = log_p[:, 0, :]
        _, selected = utils_gumbel.sample_gumbel_argmax(log_p)
        if first_action is not None:
            selected = torch.ones_like(selected) * first_action

        return log_p, selected

    def filter_empty_heaps(self, fixed, batch_state):
        self.num_of_non_empty_heaps = torch.sum(self.a_star_cpp.non_empty_heaps)

        if self.num_of_non_empty_heaps == batch_state.ids.size(0):
            return fixed, batch_state  # fixed[:batch.ids.size(0)]
        elif self.num_of_non_empty_heaps > 0:
            # if at least ine of the heaps are empty, index the input
            return fixed[self.a_star_cpp.non_empty_heaps], batch_state[self.a_star_cpp.non_empty_heaps]
        else:
            return None, None

    def init_batched_priority_queue(self, x, root_state, fixed):

        graphs_weights = torch.triu(root_state.dist, diagonal=1).view(x.size(0), -1)

        log_p, _ = self.decoder(fixed, root_state)
        log_p = log_p[:, 0, :]
        _, next_city = utils_gumbel.sample_gumbel_argmax(log_p)

        root_state = root_state.update(next_city.squeeze(-1), update_length=False)
        self.a_star_cpp.initialize(next_city, graphs_weights)

        return root_state

    def print_trajectory(self):
        t = self.a_star_cpp.getTrajectories()

        print(t.num_candidates)

    def update_py_state(self, py_state, cpp_state):
        prev_city = cpp_state.batch_prev_city[:, None]
        py_state = py_state._replace(prev_a=prev_city,
                          visited_=(~cpp_state.batch_next_actions.bool()).to(torch.uint8).unsqueeze(1),
                          cur_coord=py_state.loc[py_state.ids, prev_city],
                          i=cpp_state.batch_t.unsqueeze(1))
        return py_state

    def compute_cost(self, py_state, next_city, t, first_coord):
        next_city = next_city[:, None]
        ### compute cost (if is last city add the cost of going back to first city) ###
        cur_coord = py_state.loc[py_state.ids, next_city]
        cost = -(cur_coord - py_state.cur_coord).norm(p=2, dim=-1)

        is_done = t == py_state.loc.size(1)-2

        cost = torch.where(is_done.unsqueeze(-1),
                           cost - (first_coord - cur_coord).norm(p=2, dim=-1),
                           cost)  # add distance to first node if it's complete trajectory
        return cost

    def run_actions(self, state, actions, batch, fixed):
        outputs = []
        # state = state.to(batch.device)
        # fixed = fixed.to(batch.device)
        # self.decoder = self.decoder.to(batch.device)

        for action in actions.t():
            log_p, mask = self.decoder(fixed, state) #self._get_log_p(fixed, state)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            log_p = log_p.squeeze(1)
            action = action.long()

            state = state.update(action, update_length=True)
            # Collect output of step

            log_p = log_p.gather(1, action.unsqueeze(-1)).squeeze(-1)
            outputs.append(log_p)

        outputs = torch.stack(outputs, 1)

        return outputs
