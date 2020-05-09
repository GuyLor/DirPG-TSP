import numpy as np
import copy
import time
import torch
from torch.nn import DataParallel
import a_star_sampling
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
        state = self.encoder.problem.make_state(batch)
        fixed = self.encoder.precompute(embeddings)

        with torch.no_grad():
            opt_direct, to_log = self.sample_t_opt_search_t_direct(state,
                                                                   fixed,
                                                                   epsilon=epsilon)

        self.interactions += to_log['interactions']
        to_log['interactions'] = self.interactions

        opt, direct = zip(*opt_direct)

        trajs = self.stack_trajectories_to_batch(opt, direct,  device=batch.device)
        if trajs[1] is not None:
            opt_actions, opt_objectives = trajs[0]
            direct_actions, direct_objectives = trajs[1]

            log_p_opt, opt_length = self.run_actions(state, opt_actions, batch, fixed)
            log_p_direct, direct_length = self.run_actions(state, direct_actions, batch, fixed)

            direct_loss = (log_p_opt - log_p_direct) / epsilon
            to_log.update({'opt_cost': opt_length,
                           'direct_cost': direct_length,
                           'opt_objective': opt_objectives,
                           'direct_objective': direct_objectives,
                           'interactions': self.interactions})

        else:  # opt == direct and there's no reason for update
            direct_loss = None
            to_log.update({'opt_cost': trajs[0][0],
                           'direct_cost': trajs[0][0],
                           'opt_objective': trajs[0][1],
                           'direct_objective': trajs[0][1],
                           'interactions': self.interactions})

        return direct_loss, to_log

    def sample_t_opt_search_t_direct(self, state, fixed, epsilon=2.0, inference=False):
        start_encoder = time.time()

        batch_size = state.ids.size(0)
        # self.decoder = self.decoder.to('cpu')
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
                    batch_t.append((queue.t_opt, queue.t_direct))
                    store_stats(queue)
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
        
        print('avg interactions: ', np.mean(interactions))
        print('avg candidates: ', np.mean(candidates))
        print('avg pruned branches: ', np.mean(prune_count))

        """
        return batch_t, {j: np.sum(i)
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
        state = batch.update(selected, update_length=False)

        return log_p, state

    @staticmethod
    def stack_trajectories_to_batch(opt_traj, direct_traj, device):
        actions_opt, objectives_opt = [], []
        actions_direct, objectives_direct = [], []
        only_opt, only_opt_obj = [], []
        for t_o, t_d in zip(opt_traj, direct_traj):
            if t_o.actions != t_d.actions:
                actions_opt.append(t_o.actions)
                objectives_opt.append(t_o.objective)

                actions_direct.append(t_d.actions)
                objectives_direct.append(t_d.objective)

            else:
                only_opt.append(-t_o.length)
                only_opt_obj.append(t_o.objective)

        if actions_opt:
            opt = torch.tensor(actions_opt, device=device).split(1, 1), np.mean(objectives_opt)
            direct = torch.tensor(actions_direct, device=device).split(1, 1), np.mean(objectives_direct)
            return opt, direct

        else:
            return (torch.tensor(only_opt, device=device), np.mean(only_opt_obj)), None


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



