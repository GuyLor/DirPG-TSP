import numpy as np
import copy
import torch
from torch.nn import DataParallel
from dirpg_tsp import a_star_sampling
from utils import utils_gumbel

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

    def train_dirpg(self, batch, epsilon=1.0):
        to_log = dict.fromkeys(['interactions', 'candidates', 'prune_count', 'bfs', 'dfs', 'jumps'])
        embeddings = self.encoder(batch, only_encoder=True)
        fixed = self.encoder.precompute(embeddings)
        init_state = self.encoder.problem.make_state(batch)
        with torch.no_grad():
            _, state = self.forward_and_update(init_state, fixed)  # initialization

            pq = a_star_sampling.PriorityQueue(init_state=state,
                                               graphs=state.dist,
                                               epsilon=epsilon,
                                               search_params=self.search_params,
                                               inference=False)

            while pq.num_interactions < self.search_params.max_interactions:  ## interactions budget

                parents = pq.pop()
                prev_a = torch.gather(parents.prefix, -1, parents.t - 1)
                # prev_a = torch.where(parents.t == parents.prefix.size(-1), state.first_a, prev_a)
                visited_ = (~parents.next_actions).type(torch.uint8)

                state = state._replace(ids=parents.ids,
                                       prev_a=prev_a,
                                       visited_=visited_,  # torch.uint8
                                       lengths=-parents.lengths,
                                       cur_coord=parents.cur_coord,
                                       i=parents.t)

                log_p, state = self.forward_and_update(state, fixed)

                pq.expand(state, parents, log_p)

        self.interactions += self.search_params.max_interactions

        opt, direct = self.stack_trajectories_to_batch(pq.batch_trajectories, device=batch.device)

        opt_actions, opt_objectives = opt
        direct_actions, direct_objectives = direct

        # forward again this time gradients tracking for the backward pass
        log_p_opt, opt_length = self.run_actions(init_state, opt_actions, batch, fixed)
        log_p_direct, direct_length = self.run_actions(init_state, direct_actions, batch, fixed)

        direct_loss = (log_p_opt - log_p_direct) / epsilon
        to_log.update({'opt_cost': opt_length,
                 'direct_cost': direct_length,
                 'opt_objective': opt_objectives,
                 'direct_objective': direct_objectives,
                 'interactions': self.interactions})

        return direct_loss, to_log

    def forward_and_update(self,batch_state, fixed, first_action=None):
        log_p, _ = self.decoder(fixed, batch_state)  # fixed[:batch.ids.size(0)]
        log_p = log_p[:, 0, :]
        _, selected = utils_gumbel.sample_gumbel_argmax(log_p)
        if first_action is not None:
            selected = torch.ones_like(selected) * first_action
        state = batch_state.update(selected, update_length=True)  # like env.step
        return log_p, state

    @staticmethod
    def stack_trajectories_to_batch(batch_trajectories, device):
        """"
        batch_trajectories is a list of a_star_sampling.Trajectory objects
        inside, t_opt and t_direct objects are cpp objects with primitives
        """
        actions_opt, objectives_opt = [], []
        actions_direct, objectives_direct = [], []
        for tp in batch_trajectories:
            actions_opt.append(tp.t_opt.actions)
            actions_direct.append(tp.t_direct.actions)

            objectives_opt.append(tp.t_opt.objective)
            objectives_direct.append(tp.t_direct.objective)


        opt = torch.stack(actions_opt).to(device).split(1, 1),\
              torch.tensor(objectives_opt, device=device).mean()
        direct = torch.stack(actions_direct).to(device).split(1, 1),\
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


    """
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

        """



