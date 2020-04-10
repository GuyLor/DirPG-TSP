import numpy as np
import copy
import time
import torch
from torch.nn import DataParallel
import a_star_sampling
from utils import utils_gumbel
from scipy.sparse.csgraph import minimum_spanning_tree

class DirPG:
    def __init__(self,
                 model,
                 max_interactions=200,
                 first_improvement=False
                 ):
        model = model.module if isinstance(model, DataParallel) else model
        self.encoder = model
        self.decoder = model.decoder

        self.interactions = 0
        self.max_interactions = max_interactions
        self.first_improvement = first_improvement

    def train_dirpg(self, batch, step, epsilon=10.0):
        embeddings = self.encoder(batch, only_encoder=True)
        state = self.encoder.problem.make_state(batch)
        fixed = self.encoder.precompute(embeddings)
        # a_star_sampling.Node.epsilon = epsilon

        prune = True
        #if step % 5 == 1:
        #   self.max_interactions += 100

        if False and step > 8:
            #self.first_improvement = True
            self.max_interactions = 3000

        opt_direct, interactions = self.sample_t_opt_search_t_direct(state.to('cpu'),
                                                                     fixed.to('cpu'),
                                                                     prune=prune,
                                                                     inference=False)

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

    def sample_t_opt_search_t_direct(self, state, fixed, prune=False, inference=False):
        start_encoder = time.time()

        batch_size = state.ids.size(0)
        self.decoder = self.decoder.to('cpu')
        _, state = self.forward_and_update(state, fixed)

        queues = [a_star_sampling.PriorityQueue(init_state=state[i],
                                                distance_mat=state.dist[idx],
                                                inference=inference,
                                                max_interactions=self.max_interactions,
                                                first_improvement=self.first_improvement,
                                                prune=prune) for idx, i in enumerate(torch.tensor(range(batch_size)))]

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
                    if True and queue.id == 0:
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
        print('len interactions: ', len(interactions))
        print('mean interactions: ', np.mean(interactions))
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

    @staticmethod
    def stack_trajectories_to_batch(trajectories, device):
        actions, objectives = [], []
        for t in trajectories:
            actions.append(t.actions)
            objectives.append(t.objective)

        return torch.tensor(actions, device=device).split(1, 1), np.mean(objectives)

    @staticmethod
    def stack_lengths_to_batch(self, trajectories, device):
        return torch.tensor([t.length for t in trajectories],  device=device)

    def run_actions(self, state, actions, batch, fixed):
        outputs = []
        state = state.to(batch.device)
        fixed = fixed.to(batch.device)
        self.decoder = self.decoder.to(batch.device)
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
        outputs = torch.stack(outputs,1)

        return outputs, lengths



