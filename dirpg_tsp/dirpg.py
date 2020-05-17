import numpy as np
import copy
import torch
from torch.nn import DataParallel
from dirpg_tsp import a_star_sampling
from utils import utils_gumbel

class DirPG:
    def __init__(self,
                 model,
                 opts
                 ):
        model = model.module if isinstance(model, DataParallel) else model
        self.encoder = model
        self.decoder = model.decoder

        self.interactions = 0

        self.search_params = {'max_interactions': opts.max_interactions,
                              'first_improvement': opts.first_improvement,
                              'dfs_like': opts.dfs_like,
                              'independent_gumbel': opts.independent_gumbel,
                              'heuristic': opts.heuristic,
                              'alpha': opts.alpha,
                              'dynamic_weighting': opts.dynamic_weighting,
                              'prune': not opts.not_prune,
                              'epsilon': opts.epsilon}

    def train_dirpg(self, batch, epsilon=1.0):
        embeddings = self.encoder(batch, only_encoder=True)
        fixed = self.encoder.precompute(embeddings)
        state = self.encoder.problem.make_state(batch)

        # TODO: create cpp Batchedstate object


        with torch.no_grad():
            opt_direct, to_log = self.sample_t_opt_search_t_direct(state,
                                                                   fixed,
                                                                   epsilon=epsilon)

        self.interactions += to_log['interactions']

        opt, direct = zip(*opt_direct)

        opt, direct = self.stack_trajectories_to_batch(opt, direct,  device=batch.device)

        opt_actions, opt_objectives = opt
        direct_actions, direct_objectives = direct

        log_p_opt, opt_length = self.run_actions(state, opt_actions, batch, fixed)
        log_p_direct, direct_length = self.run_actions(state, direct_actions, batch, fixed)

        direct_loss = (log_p_opt - log_p_direct) / (10.+epsilon)
        to_log.update({'opt_cost': opt_length,
                       'direct_cost': direct_length,
                       'opt_objective': opt_objectives,
                       'direct_objective': direct_objectives,
                       'interactions': self.interactions})

        return direct_loss, to_log

    def sample_t_opt_search_t_direct(self, state, fixed, epsilon=2.0, inference=False):

        batch_size = state.ids.size(0)
        _, state = self.forward_and_update(state, fixed)

        queues = [a_star_sampling.PriorityQueue(init_state=state[i],
                                                distance_mat=state.dist[idx],
                                                epsilon=epsilon,
                                                inference=inference,
                                                search_params=self.search_params)
                  for idx, i in enumerate(torch.tensor(range(batch_size)))]

        batch_t, interactions = [], []
        candidates, prune_count = [], []
        bfs, dfs, jumps = [], [], []

        def store_stats(q):
            candidates.append(len(q.trajectories_list))
            prune_count.append(q.prune_count)
            interactions.append(q.num_interactions)
            bfs.append(q.bfs)
            dfs.append(q.dfs)
            jumps.append(q.others)

        while queues:  # batch
            parents = []
            copy_queues = copy.copy(queues)
            for queue in copy_queues:
                parent = queue.pop()

                if parent == 'break':
                    batch_t.append((queue.t_opt, queue.t_direct))
                    store_stats(queue)
                    queues.remove(queue)
                    continue
                else:
                    parents.append(parent)

            if len(parents) > 0:
                batch_state = state.stack_state(parents)
                log_p, state = self.forward_and_update(batch_state, fixed)

                # TODO: make sure that log_p is not on the gpu before .numpy()
                log_p = log_p.numpy()

                idx = torch.tensor(range(len(queues)))
                for i, queue in zip(idx, queues):
                    queue.expand(state[i], log_p[i])

        return batch_t, {j: np.sum(i) if j == 'interactions' else np.mean(i)
                         for i, j in zip([interactions, candidates, prune_count, bfs, dfs, jumps],
                                         ['interactions', 'candidates', 'prune_count', 'bfs', 'dfs', 'jumps'])}

    def forward_and_update(self, batch, fixed, first_action=None):

        self.decoder.eval()
        log_p, _ = self.decoder(fixed[:batch.ids.size(0)], batch)
        log_p = log_p[:, 0, :]
        _, selected = utils_gumbel.sample_gumbel_argmax(log_p)
        if first_action is not None:
            selected = torch.ones_like(selected) * first_action
        #selected = torch.argmax(log_p, -1)
        state = batch.update(selected, update_length=True)


        # special_action = state.prev_a.item()   ----->> selected ?
        # s = time.time()
        # not_visited = [i for i in self.current_node.not_visited if i != special_action] ----->> mask ?
        #length = -(cur_coord - self.current_node.cur_coord).norm(p=2, dim=-1)
        # cur_coord = state.loc[self.current_node.id, special_action]
        # length = state.lengths
        # if len(self.current_node.prefix)+1 == self.graph_size:
            # length -= (self.first_coord - cur_coord).norm(p=2, dim=-1)


        return log_p, state

    @staticmethod
    def stack_trajectories_to_batch(opt_traj, direct_traj, device):
        actions_opt, objectives_opt = [], []
        actions_direct, objectives_direct = [], []
        for t_o, t_d in zip(opt_traj, direct_traj):
            actions_opt.append(t_o.actions)
            objectives_opt.append(t_o.objective)

            actions_direct.append(t_d.actions)
            objectives_direct.append(t_d.objective)

        opt = torch.tensor(actions_opt, device=device).split(1, 1),\
              torch.tensor(objectives_opt, device=device).mean()
        direct = torch.tensor(actions_direct, device=device).split(1, 1),\
                 torch.tensor(objectives_direct, device=device).mean()
        return opt, direct


    @staticmethod
    def stack_lengths_to_batch(trajectories, device):
        return torch.tensor([t.length for t in trajectories],  device=device)

    def run_actions(self, state, actions, batch, fixed):
        outputs = []
        # state = state.to(batch.device)
        # fixed = fixed.to(batch.device)
        # self.decoder = self.decoder.to(batch.device)
        for action in actions:
            log_p, mask = self.decoder(fixed, state) #self._get_log_p(fixed, state)
            # Select the indices of the next nodes in the sequences, result (batch_size) long
            log_p = log_p.squeeze(1)

            state = state.update(action.squeeze(1), update_length=True)
            # Collect output of step
            log_p = log_p.gather(1, action).squeeze(-1)
            outputs.append(log_p)

        # Collected lists, return Tensor
        a = torch.cat(actions, 1)
        d = batch.gather(1, a.unsqueeze(-1).expand_as(batch))

        lengths = state.lengths + (d[:, 0] - d[:, -1]).norm(p=2, dim=1).unsqueeze(-1)
        outputs = torch.stack(outputs, 1)

        return outputs, lengths



