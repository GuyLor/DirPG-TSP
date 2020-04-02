import numpy as np
import copy
import time
import torch
import heapq
from utils import utils_gumbel
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
    epsilon = 10.0

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
                 max_gumbel=None,
                 t_opt=True,
                 t_opt_objective=None,
                 dfs_like=False):  # How much total objective t_opt achieved.

        if max_gumbel is None:
            max_gumbel = utils_gumbel.sample_gumbel(0)

        self.id = id
        self.first_a = first_a

        self.not_visited = not_visited
        self.prefix = prefix
        self.t = len(self.prefix)

        self.lengths = lengths

        self.cur_coord = cur_coord

        self.t_opt_objective = t_opt_objective
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
        return self.get_objective()

    def get_priority_max_gumbel(self):
        return self.max_gumbel

    def get_upper_bound(self):
        return self.max_gumbel + self.epsilon * (self.lengths + self.bound_length_togo())

    def bound_length_togo(self):
        # TODO calculate the MST upper bound for TSP
        return 0

    def get_objective(self):
        """Computes the objective of the trajectory.
        Only used if a node is terminal.
        """
        return self.max_gumbel + self.epsilon * self.lengths

    def print(self):
        att = ['id','first_a','prefix','t','lengths','t_opt_objective','done','logprob_so_far','max_gumbel','next_actions','t_opt']
        print(' -----------    Node     -----------')
        print('id:  ', self.id)
        print('first_a:  ', self.first_a)
        print('prefix:  ', self.prefix)
        print('not visited:  ', self.not_visited)
        print('next_actions:  ', self.next_actions)
        print('t:  ', self.t)
        print('lengths:  ', self.lengths)
        print('t_opt_objective:  ', self.t_opt_objective)
        print('done:  ', self.done)
        print('logprob_so_far:  ', self.logprob_so_far)
        print('max_gumbel:  ', self.max_gumbel)
        print('t_opt:  ', self.t_opt)
        print(' -------------------------------')


class PriorityQueue:
    def __init__(self,
                 init_state,
                 inference=False,
                 max_search_time=1000,
                 max_interactions=500,
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
                         next_actions=torch.tensor(not_visited),  # number of cities
                         not_visited=not_visited,
                         prefix=[special_action],
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
        self.max_search_time = max_search_time
        self.num_interactions = 0
        self.max_interactions = max_interactions
        self.dfs_like = dfs_like
        self.inference = inference

    def pop(self):
        parent = heapq.heappop(self.queue)
        self.current_node = parent
        self.t_opt_objective = self.t_opt.objective if self.t_opt is not None else None
        self.lower_bound_objective = -float('Inf') if self.t_direct is None else self.t_direct.objective

        if False and self.lower_bound_objective > parent.upper_bound:
            self.prune_count += 1
            return self.pop()

        # Start the search time count
        if not parent.t_opt and not self.start_search_direct:
            self.start_time = time.time()
            self.start_search_direct = True

        if parent.done:
            return self.set_trajectory(parent)

        if time.time() - self.start_time > self.max_search_time or self.num_interactions >= self.max_interactions:
            # print("*****  time's-up/max interactions   *****")
            parent = 'break'

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
            if self.inference:
                return 'break'
        else:
            if t.objective > self.t_direct.objective:
                self.t_direct = t
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
        length = - (cur_coord - self.current_node.cur_coord).norm(p=2, dim=-1)

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
            t_opt=self.current_node.t_opt,
            t_opt_objective=self.t_opt_objective,
            dfs_like=self.dfs_like)

        if special_child.upper_bound < self.lower_bound_objective:
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
                t_opt=False,
                t_opt_objective=self.t_opt_objective,
                dfs_like=False)

            if other_children.upper_bound < self.lower_bound_objective:
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

        self.encoder = model
        self.decoder = model.decoder

    def train_dirpg(self, batch, epsilon=1.0):
        embeddings = self.encoder(batch, only_encoder=True)
        state = self.encoder.problem.make_state(batch)
        fixed = self.encoder.precompute(embeddings)

        opt_direct = self.sample_t_opt_search_t_direct(state, fixed, inference=False)
        opt, direct = zip(*opt_direct)

        opt_actions = self.stack_trajectories_to_batch(opt)
        direct_actions = self.stack_trajectories_to_batch(direct)

        log_p_opt, opt_length = self.run_actions(state, opt_actions, batch, fixed)
        log_p_direct, direct_length = self.run_actions(state, direct_actions, batch, fixed)

        out = (log_p_opt - log_p_direct)/(epsilon+1e-7)

        return out, (opt_length, direct_length)

    def sample_t_opt_search_t_direct(self, state, fixed, inference=False):
        start_encoder = time.time()

        batch_size = state.ids.size(0)
        _, state = self.forward_and_update(state, fixed)
        queues = [PriorityQueue(state[i],
                                inference=inference) for i in torch.tensor(range(batch_size))]

        batch_t = []
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
                    queues.remove(queue)
                    continue
                else:
                    nodes.append(parent)
            after_pop = time.time()
            if len(nodes) > 0:

                batch_state = state.stack_state(nodes)
                after_stack = time.time()
                log_p, state = self.forward_and_update(batch_state,fixed)

                log_p = log_p.numpy()
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
        return batch_t

    def forward_and_update(self, batch, fixed):
        with torch.no_grad():
            self.decoder.eval()
            log_p, _ = self.decoder(fixed, batch)

        log_p = log_p[:, 0, :]

        #_, selected = utils_gumbel.sample_gumbel_argmax(log_p)
        selected = torch.argmax(log_p, -1)
        state = batch.update(selected, update_length=False)

        return log_p, state

    def stack_trajectories_to_batch(self, nodes):
        return torch.tensor([node.actions for node in nodes]).split(1,1)

    def stack_lengths_to_batch(self, nodes):
        return torch.tensor([node.length for node in nodes])

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




"""
class DirectAstar(MinigridRL):
    def __init__(self,
                 env_path,
                 chekpoint,
                 seed,
                 independent_sampling = False,
                 keep_searching = False,
                 max_steps=240,
                 discount=0.99,
                 alpha=0.2,
                 max_search_time=30,
                 max_interactions = 3000,
                 mixed_search_strategy = False,
                 optimization_method = 'direct',
                 dfs_like=False,
                 eps_grad=1.0,
                 eps_reward=3.0):
        super().__init__(env_path,chekpoint,seed,max_steps,max_interactions,discount)
        
        self.max_search_time=max_search_time
        self.max_interactions = max_interactions
        self.independent_samplimg = independent_sampling
        self.dfs_like=dfs_like
        self.mixed_search_strategy = mixed_search_strategy
        self.max_steps=max_steps
        self.alpha=alpha,
        self.eps_grad = eps_grad
        self.eps_reward = eps_reward
        Node.epsilon = eps_reward
        Node.discount = self.discount
        Node.alpha = self.alpha[0]
        self.break_on_goal = True
        # Node.doors_togo = len(self.env.rooms)-1
        self.max_interactions = self.max_steps*30
        self.keep_searching = keep_searching # if true, keep searching for t_direct even if priority(t_direct)>priority(t_opt)
        self.optimization_method = optimization_method
        self.log['priority_func'] = inspect.getsource(Node)
        self.log['search_proc'] = inspect.getsource(DirectAstar)

    def sample_t_opt_and_search_for_t_direct(self,inference=False,to_print=False):
        self.env.seed(self.seed)
        root_node = Node(
            env = self.env,
            states = [self.env.reset()],
            max_gumbel=utils.sample_gumbel(0),
            next_actions=range(self.num_actions),
            t_opt=True)

        queue = []
        heapq.heappush(queue,root_node)
        
        final_trajectories = []
        start_time = float('Inf')
        start_search_direct = False
        prune_count =0
        num_interactions=0
        dfs_like = self.dfs_like
        t_opt = None  # Will get set when we find t_opt.
        t_direct = None  # Will get set when we find t_opt.
        while queue:

            parent = heapq.heappop(queue)
            t_opt_objective = t_opt.node.objective if t_opt is not None else None
            lower_bound_objective = -float('Inf') if t_direct is None else t_direct.node.objective

            if lower_bound_objective > parent.upper_bound:
                prune_count += 1
                continue

            #Start the search time count
            if not parent.t_opt and not start_search_direct:
                start_time = time.time()
                start_search_direct = True

            if parent.done:
                status = True if parent.undiscounted_reward == Node.goal_reward else False
                t = Trajectory(actions=parent.prefix,
                               states=parent.states,
                               gumbel=parent.max_gumbel,
                               reward=parent.reward_so_far,
                               status=status,
                               node = parent)

                assert len(t.actions) == len(parent.states)-1
                final_trajectories.append(t)
                if parent.t_opt:
                    t_opt = t
                    t_direct = t
                else:

                    if t.node.objective > t_direct.node.objective:
                        t_direct = t
                        if not self.keep_searching:
                            print('*****  priority(direct) > priority(opt)   *****')
                            break
                if t.status and self.break_on_goal:
                    print('*****  GOAL BREAK   *****')
                    break
                continue
            
            if time.time()-start_time>self.max_search_time or num_interactions >= self.max_interactions:
                #print("*****  time's-up/max interactions   *****")
                break

            current_state = parent.states[-1]
            with torch.no_grad():
                self.policy.eval()
                action_logprobs = self.policy([current_state]).cpu().numpy().squeeze(0)
            next_action_logprobs = action_logprobs[parent.next_actions]
            maxval,special_action_index = utils.sample_gumbel_argmax(next_action_logprobs)


            special_action = parent.next_actions[special_action_index]
            special_action_logprob = action_logprobs[special_action]

            env_copy = copy.deepcopy(parent.env) # do it here, before the step
            new_state,reward,done,info = parent.env.step(special_action)
            num_interactions += 1

            special_child = Node(
                                 env = parent.env,
                                 prefix=parent.prefix + [special_action],
                                 states=parent.states + [new_state],
                                 parent_reward_so_far=parent.reward_so_far,
                                 undiscounted_reward=reward,
                                 rewards_list=parent.rewards_list + [reward],
                                 done=done,
                                 logprob_so_far=parent.logprob_so_far + special_action_logprob,
                                 max_gumbel=parent.max_gumbel,
                                 next_actions=range(self.num_actions),# All next actions are possible.
                                 parent_doors_togo=parent.doors_togo,
                                 t_opt = parent.t_opt,
                                 t_opt_objective=t_opt_objective,
                                 dfs_like = dfs_like)
                                 
            if special_child.upper_bound < lower_bound_objective:
                prune_count+=1
                continue
            else:
                heapq.heappush(queue,special_child)
            
            # Sample the max gumbel for the non-chosen actions and create an "other
            # children" node if there are any alternatives left.
            other_actions = [i for i in parent.next_actions if i != special_action]
            assert len(other_actions) == len(parent.next_actions) - 1
            if other_actions:
                other_max_location = utils.logsumexp(action_logprobs[other_actions])
                other_max_gumbel = utils.sample_truncated_gumbel(parent.logprob_so_far + other_max_location,parent.max_gumbel)
                other_children = Node(
                                    env = env_copy,
                                    prefix=parent.prefix,
                                    states=parent.states,
                                    parent_reward_so_far=parent.reward_so_far,
                                    rewards_list=parent.rewards_list,
                                    done=parent.done,
                                    logprob_so_far=parent.logprob_so_far,
                                    max_gumbel=other_max_gumbel,
                                    next_actions=other_actions,
                                    parent_doors_togo=parent.doors_togo,
                                    t_opt = False,
                                    t_opt_objective=t_opt_objective,
                                    dfs_like = False)

                if other_children.upper_bound < lower_bound_objective:
                    prune_count+=1
                    continue
                else:
                    heapq.heappush(queue,other_children)
        if not inference:
            print ('pruned branches: {},'
                   ' t_direct candidates: {},'
                   ' nodes left in queue: {},'
                   ' num interactions: {} '.format(prune_count,len(final_trajectories), len(queue),num_interactions))

        return t_opt, t_direct,final_trajectories,num_interactions
        
    def get_one_side_loss(self,t,is_direct):
        states  = t.states[:-1]
        actions = torch.LongTensor(t.actions).view(-1,1)
        
        phi = self.policy(states) # gets the logits so the network will calculates weights gradients
        
        if is_direct:
            y = -torch.FloatTensor(actions.size(0),phi.size(1)).zero_().scatter_(-1,actions,1.0)
        else:
            y =  torch.FloatTensor(actions.size(0),phi.size(1)).zero_().scatter_(-1,actions,1.0)
        
        y_opt_direct = utils.use_gpu(y)
        policy_loss = torch.sum(y_opt_direct*phi)/self.eps_grad
        return policy_loss
                
    def cross_entropy_loss(self,final_trajectories, elite_frac = 0.05):
        final_trajectories.sort(key=lambda x: x.reward, reverse=True)
        end = math.ceil(len(final_trajectories)*elite_frac)
        print ('len(final_trajectories): ',len(final_trajectories[:end]))
        ce_loss = 0
        for t in final_trajectories[:end]:
            ce_loss += self.get_one_side_loss(t,is_direct=True)
        return ce_loss

    def train(self, num_episodes=500, seed = 1234):
        
        self.seed = seed
        rewards_opt_direct = []
        priority_opt_direct = []
        lengths_opt_direct=[]
        interactions = []
        candidates = []
        to_plot_opt = []
        to_plot_direct = []
        self.log['start_seed'] = seed
        total_interactions = 6e6
        count_interactions=0
        episode=0
        
        sampling = self.sample_trajectories if self.independent_samplimg else self.sample_t_opt_and_search_for_t_direct
        while count_interactions < self.max_interactions*num_episodes: #total_interactions:
            self.env.seed(self.seed)
            episode+=1
            
            print('--------- new map {} -------------'.format(episode))
            t_opt, t_direct,final_trajectories,num_interactions = sampling()

            for i in range(1):
                self.policy.train()
                if self.optimization_method == 'direct':
                    policy_loss = self.direct_optimization_loss(t_opt, t_direct)
                elif self.optimization_method == 'CE':
                    policy_loss = self.cross_entropy_loss(final_trajectories, elite_frac = 0.05)
                    
                print(i,policy_loss)
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

            interactions.append(num_interactions)

            opt_reward = t_opt.reward
            direct_reward=t_direct.reward
            print ('opt reward: {:.3f}, success: {}, length: {}, priority: {:.3f}, objective: {:.3f}'.format(opt_reward, t_opt.status, len(t_opt.actions),t_opt.node.priority,t_opt.node.objective))
            print ('direct reward: {:.3f}, success: {}, length: {}, priority: {:.3f}, objective: {:.3f}'.format(direct_reward,t_direct.status, len(t_direct.actions),t_direct.node.priority,t_direct.node.objective))
            
            to_plot_opt.append((count_interactions+len(t_opt.actions),opt_reward))
            to_plot_direct.append((count_interactions+num_interactions,direct_reward))
            candidates.append(len(final_trajectories))
            rewards_opt_direct.append((opt_reward,direct_reward))
            lengths_opt_direct.append((len(t_opt.actions),len(t_direct.actions)))
            priority_opt_direct.append((t_opt.node.priority,t_direct.node.priority))
            sys.stdout.flush()
            count_interactions+=num_interactions
            self.seed+=1
            if episode % 20 == 1:
                self.save_checkpoint()
            
        self.save_checkpoint()
        self.log['interactions']=interactions
        self.log['num_candidates'] = candidates
        self.log['rewards_opt_direct'] = rewards_opt_direct
        self.log['lengths_opt_direct'] = lengths_opt_direct
        self.log['priority_opt_direct'] = priority_opt_direct
        
        self.log['to_plot_opt'] =to_plot_opt
        self.log['to_plot_direct'] =to_plot_direct
        
        return rewards_opt_direct
        
    def collect_data_candidates_direct_returns(self,exps=40,alphas=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
        sampling = self.sample_trajectories if self.independent_samplimg else self.sample_t_opt_and_search_for_t_direct
        
        returns_of_candidates_alphas,direct_obj_of_candidates_alphas,direct_priority_of_candidates_alphas = {},{},{}
        self.break_on_goal = False
        for alpha in alphas:
            direct_returns_of_all_candidates = []
            direct_obj = []
            direct_priority =[]
            Node.alpha = alpha
            seed = self.seed
            seed += 100000
            print ('-'*100,'\n',alpha,'\n','-'*100)
            for exp in range(exps):
                self.env.seed(seed)
                #print('--------- new map {} -------------'.format(seed))
                t_opt, t_direct,final_trajectories,num_interactions = sampling()
                seed+=1
                direct_returns_of_all_candidates += [t.reward for t in final_trajectories]
                direct_obj += [t.node.objective for t in final_trajectories]
                direct_priority += [t.node.priority for t in final_trajectories]
            
            returns_of_candidates_alphas[alpha] = direct_returns_of_all_candidates
            direct_obj_of_candidates_alphas[alpha] = direct_obj
            direct_priority_of_candidates_alphas[alpha] = direct_priority
        
        self.log['returns_of_candidates']=returns_of_candidates_alphas
        self.log['direct_obj_of_candidates']=direct_obj_of_candidates_alphas
        self.log['direct_priority_of_candidates']=direct_priority_of_candidates_alphas
        self.break_on_goal = True
"""