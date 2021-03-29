import numpy as np
import torch
from torch.nn import DataParallel
from utils import utils_gumbel, functions
import dirpg_cpp
import copy
#import io
#from concorde.tsp import TSPSolver
#import collections as C


def print_cpp_state(state_cpp):
    print("=============  cpp state ============ ")
    print("prev_city: ",state_cpp.batch_prev_city)
    print(state_cpp.batch_prev_city.size())
    print("t: ", state_cpp.batch_t)
    print(state_cpp.batch_t.size())
    #print("done: ", state_cpp.done)
    print("next_actions: ")
    print(state_cpp.batch_next_actions)
    print(state_cpp.batch_next_actions.size())
    print("==================================== ")

def print_trajectory(t_opt,t_direct,t):
    print("------- trajectory ----------")
    print(t_opt.actions, t_opt.actions.size() )
    print(t_direct.actions, t_direct.actions.size())
    print(t.num_candidates, len(t.num_candidates))
    print("------- ----- ----------")

class DirPG:
    def __init__(self,
                 model,
                 search_params
                 ):
        model = model.module if isinstance(model, DataParallel) else model
        self.model = model
        self.decoder = model.decoder
        self.model.set_decode_type('sampling')
        self.decoder.count_interactions = False
        self.interactions = 0
        self.search_params = search_params
        priority = 0 if search_params.gumbel_top_k else -1

        if not search_params.supervised:
            self.a_star_cpp = dirpg_cpp.AstarSampling(search_params.batch_size,
                                                      search_params.max_interactions,
                                                      search_params.graph_size,
                                                      search_params.epsilon,
                                                      search_params.alpha,
                                                      search_params.dynamic_weighting,
                                                      priority,
                                                      search_params.not_prune,
                                                      search_params.dfs_like,
                                                      search_params.optimal_heuristic,
                                                      search_params.k_improvement,
                                                      7)


        self.iteration = 0
        self.sum_avg = [0, 0, 0, 0, 0, 0, 0, 0 ]
        self.to_log = dict.fromkeys(['opt_cost',
                                     'direct_cost',
                                     'opt_objective',
                                     'direct_objective',
                                     'empty_heaps',
                                     'prune_count',
                                     'candidates'])
        self.denom = 0

    def train_dirpg(self, batch, budget, epsilon=1.0):
        to_log = dict.fromkeys(['interactions', 'candidates', 'prune_count', 'bfs', 'dfs', 'jumps'])
        self.num_of_non_empty_heaps = batch.size(0)
        self.a_star_cpp.clear()
        embeddings = self.model(batch, only_encoder=True)
        init_fixed = self.model.precompute(embeddings)
        init_state = self.model.problem.make_state(batch)
        #empty_heaps = torch.tensor([True for _ in range(batch.size(0))])
        interactions_b = copy.copy(self.interactions)
        #print("*"*60)

        with torch.no_grad():
            py_state, fixed = self.init_batched_priority_queue(batch, init_state, init_fixed)
            self.decoder.count_interactions = True
            self.first_coord = py_state.loc[py_state.ids, py_state.first_a, :]
            step = 0
            #empty_heaps = self.a_star_cpp.getEmptyHeapsCount()
            while step < budget:  ## interactions budget
                # print("^^^^^^^^^^^^   step==", step, "   ^^^^^^^^^^^^^^^")
                cpp_state = self.pop_batch()

                py_state = self.update_py_state(py_state, cpp_state)

                fixed, py_state = self.filter_empty_heaps(fixed, py_state)
                self.interactions += self.num_of_non_empty_heaps

                if self.num_of_non_empty_heaps == 0:
                    #print("stop!!   self.num_of_non_empty_heaps == 0")
                    break
                log_p, next_city = self.decode_and_sample(fixed, py_state)

                cost = self.compute_cost(py_state, next_city, cpp_state)

                self.expand_heaps(next_city, log_p, cost)

                step += 1

                #self.search_params.max_interactions
        self.decoder.count_interactions = False
        t = self.a_star_cpp.getTrajectories()
        t_opt, t_direct = t.get_t_opt_direct()

        # forward again this time gradients tracking for the backward pass

        log_p_opt = self.model.run_actions(init_state, t_opt.actions, batch, init_fixed)  #, opt_length
        log_p_direct = self.model.run_actions(init_state, t_direct.actions, batch, init_fixed)  #, direct_length

        # direct_loss = (log_p_opt - log_p_direct) #/epsilon #*np.sign(epsilon)
        direct_loss = torch.cat((log_p_opt, -log_p_direct), 0)/epsilon
        if self.iteration % int(self.search_params.log_step) == 0:
            print('self.interactions: ', self.interactions)
            to_log.update({'opt_cost': np.mean(t_opt.costs),
                           'direct_cost': np.mean(t_direct.costs),
                           'opt_objective':np.mean(t_opt.objectives),
                           'direct_objective': np.mean(t_direct.objectives),
                           'empty_heaps': self.a_star_cpp.getEmptyHeapsCount(),
                           'prune_count': np.mean(t.prune_count),
                           'candidates': np.mean(t.num_candidates),
                           'interactions_count': float(self.interactions-interactions_b)/self.search_params.batch_size,
                           'interactions': self.interactions})

        self.iteration += 1
        return direct_loss, to_log

    def pop_batch(self):
        return self.a_star_cpp.popBatch()

    def expand_heaps(self, next_city, log_p, cost):
        self.a_star_cpp.expand(next_city.view(-1).cpu().tolist(),
                               log_p.view(next_city.size(0),-1).cpu().tolist(),
                               cost.view(-1).cpu().tolist())


    def decode_and_sample(self, fixed, batch_state, first_action=None):

        log_p, _ = self.decoder(fixed, batch_state, True)

        log_p = log_p[:, 0, :]
        _, selected = utils_gumbel.sample_gumbel_argmax(log_p)
        if first_action is not None:
            selected = torch.ones_like(selected) * first_action

        return log_p, selected

    def filter_empty_heaps(self, fixed, batch_state):
        self.non_empty_heaps = self.a_star_cpp.getNonEmptyHeaps().to(self.search_params.device)
        self.num_of_non_empty_heaps = torch.sum(self.non_empty_heaps).item()

        if self.num_of_non_empty_heaps == batch_state.ids.size(0):
            out = fixed, batch_state  # fixed[:batch.ids.size(0)]
        elif self.num_of_non_empty_heaps > 0:
            # if at least ine of the heaps are empty, index the input
            #print("5r5r5r5r5r5r5     ", self.num_of_non_empty_heaps, "    r5r5r5r5r5r5r5r")
            self.first_coord = self.first_coord[self.non_empty_heaps]
            out = fixed[self.non_empty_heaps], batch_state[self.non_empty_heaps]
        else:
            #print("non_empty_heaps = ", self.num_of_non_empty_heaps)
            #print("return None, None")
            out = None, None
        self.a_star_cpp.updateHeapFilter()
        return out

    def init_batched_priority_queue(self, x, root_state, fixed):
        if self.search_params.optimal_heuristic:
            graphs_weights = root_state.dist.view(x.size(0), -1)
        else:
            graphs_weights = torch.triu(root_state.dist, diagonal=1).view(x.size(0), -1)

        log_p, mask = self.decoder(fixed, root_state)
        next_city = self.model._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
        #_, next_city = utils_gumbel.sample_gumbel_argmax(log_p)

        next_city = next_city.squeeze(-1) if next_city.size(0) > 1 else next_city
        root_state = root_state.update(next_city, update_length=True)
        self.a_star_cpp.initialize(next_city.cpu().tolist(), graphs_weights.cpu().tolist())
        return root_state, fixed

    def print_trajectory(self):
        t = self.a_star_cpp.getTrajectories()
        print(t.num_candidates)

    def update_py_state(self, py_state, cpp_state):
        prev_city = cpp_state.batch_prev_city[:, None]
        py_state = py_state._replace(prev_a=prev_city.to(self.search_params.device),
                                     visited_=(~cpp_state.batch_next_actions.bool()).to(torch.uint8).to(self.search_params.device).unsqueeze(1),
                                     cur_coord=py_state.loc[py_state.ids, prev_city].to(self.search_params.device),
                                     i=cpp_state.batch_t.unsqueeze(1).to(self.search_params.device))
        return py_state

    def compute_cost(self, py_state, next_city, state_cpp):
        next_city = next_city[:, None]
        ### compute cost (if is last city add the cost of going back to first city) ###
        cur_coord = py_state.loc[py_state.ids, next_city]
        cost = (cur_coord - py_state.cur_coord).norm(p=2, dim=-1)

        is_done = (state_cpp.batch_t == py_state.loc.size(1)-2).to(self.search_params.device)

        cost = torch.where(is_done[self.non_empty_heaps].unsqueeze(-1),
                           cost + (self.first_coord - cur_coord).norm(p=2, dim=-1),
                           cost)  # add distance to first node if it's complete trajectory

        return cost

    def train_with_concorde(self, batch):
        from problems.tsp import concorde_solver
        embeddings = self.model(batch, only_encoder=True)
        init_fixed = self.model.precompute(embeddings)
        init_state = self.model.problem.make_state(batch)

        # sampling the first city (not changing the TSP)
        self.decoder.count_interactions = True
        log_p, mask = self.decoder(init_fixed, init_state)
        # Select the indices of the next nodes in the sequences, result (batch_size) long
        first_cities = self.model._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
        #_, next_city = utils_gumbel.sample_gumbel_argmax(log_p)


        state = init_state.update(first_cities, update_length=True)
        with torch.no_grad():
            #self.model.eval()
            _log_p, opt_actions = self.model._inner(batch, embeddings, init_state=state)
            #print(opt_actions)
            opt_costs, mask = self.model.problem.get_costs(batch, opt_actions)
            log_p = torch.cat((log_p, _log_p), 1)
            ll = self.model._calc_log_likelihood(log_p, opt_actions, mask)

        t = concorde_solver.solve_batch_graphs(batch, first_cities, self.search_params.run_name[-6:])
        self.decoder.count_interactions = True
        log_p_direct = self.model.run_actions(init_state, t.actions, batch, init_fixed)

        #self.model.train()
        #log_p_opt = self.run_actions(init_state, opt_actions, batch, init_fixed)

        #direct_loss = (log_p_opt - log_p_direct)
        direct_loss = - log_p_direct

        #self.interactions += 2*batch.size(0)*batch.size(1)
        to_log = {}

        if self.iteration % int(self.search_params.log_step) == 0:
            to_log.update({'opt_cost': opt_costs.mean(),
                           'direct_cost': np.mean(t.costs),
                           'opt_objective': - ll.mean().cpu().item(),
                           'direct_objective': direct_loss.mean().cpu().item(),
                           'interactions': self.interactions})

        self.iteration += 1

        return direct_loss, to_log

    def train_with_concorde_(self, batch):
        embeddings = self.model(batch, only_encoder=True)
        init_fixed = self.model.precompute(embeddings)
        init_state = self.model.problem.make_state(batch)

        first_cities = torch.randint(low=0, high=batch.size(1), size=(batch.size(0),)).to(batch.device)
        #_, next_city = utils_gumbel.sample_gumbel_argmax(log_p)
        first_cities = first_cities.squeeze(-1) if first_cities.size(0) > 1 else first_cities

        init_state = init_state.update(first_cities, update_length=True)
        with torch.no_grad():
            self.model.eval()
            _, opt_actions = self.model._inner(batch, embeddings, init_state=init_state)
            #print(opt_actions)
            opt_costs, _ = self.model.problem.get_costs(batch, opt_actions)

        f = io.BytesIO()
        #with functions.stdout_redirector(f):
        t = solve_batch_graphs(batch, first_cities, self.search_params.run_name[-5:])
        self.model.train()
        log_p_direct = self.run_actions(init_state, t.actions, batch, init_fixed)

        #self.model.train()
        #log_p_opt = self.run_actions(init_state, opt_actions, batch, init_fixed)

        #direct_loss = (log_p_opt - log_p_direct)
        direct_loss = - log_p_direct

        self.interactions += batch.size(0)*batch.size(1)
        to_log = {}
        self.sum_avg[0] += opt_costs.mean()
        self.sum_avg[1] += np.mean(t.costs)
        #self.sum_avg[2] += direct_loss.cpu().item()
        self.sum_avg[3] += direct_loss.mean().cpu().item()
        self.denom += 1
        if self.iteration % int(self.search_params.log_step) == 0:
            to_log.update({'opt_cost': self.sum_avg[0] /self.denom,
                           'direct_cost': self.sum_avg[1] /self.denom,
                           'opt_objective': self.sum_avg[2] / self.denom,
                           'direct_objective': self.sum_avg[3] / self.denom,
                           'interactions': self.interactions})

        self.iteration += 1

        return direct_loss, to_log