import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
                i=self.i[key],
            )
        return super(StateTSP, self).__getitem__(key)

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateTSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(batch_size, 1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected, update_length=True):
        # Update the state
        #prev_a = selected[:, None]  # Add dimension for step
        prev_a = selected[:, None]
        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]

        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if update_length and self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = torch.where(self.i ==torch.zeros(lengths.size(),
                                                   dtype=torch.int64,
                                                   device=self.loc.device), prev_a, self.first_a)
        #first_a = prev_a if self.i[0].item() == 0 else self.first_a
        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps

        return (self.i >= self.loc.size(-2)).data.any()

    def get_current_node(self):

        return self.prev_a

    def get_mask(self):
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions

    def print_size(self):
        for s in self._fields:
            print(s,':  ', getattr(self, s).size()) if getattr(self, s) is not None else print(None)
            #print(getattr(self, s).size())

    def print_state(self):
        print('----------   state ----------')
        for s in self._fields:
            if s == 'loc' or s == 'dist':
                continue
            print(s)
            #print(getattr(self, s).size())
            print(getattr(self, s))
        print('----------------------------')

    def to(self, device):
        return self._replace(
            loc=self.loc.to(device),
            dist=self.dist.to(device),
            ids=self.ids.to(device),
            first_a=self.first_a.to(device),
            prev_a=self.prev_a.to(device),
            visited_=self.visited_.to(device),
            lengths=self.lengths.to(device),
            cur_coord=self.cur_coord.to(device) if self.cur_coord is not None else None,
            i=self.i.to(device)
        )


    def stack_state(self, nodes_list):
        ids, first_a, prev_a, cur_coord, visited_, lengths, i = [], [], [], [], [], [], []
        for node in nodes_list:
            ids.append(node.id)
            first_a.append(node.first_a)
            prev_a.append([node.prefix[-1]] if len(node.prefix) > 0 else [node.first_a])
            mask = [0 if i in node.next_actions else 1 for i in range(self.loc.size(1))]
            visited_.append([mask])
            lengths.append(-node.lengths)
            cur_coord.append(node.cur_coord)
            i.append(node.t)

        prev_a = torch.tensor(prev_a, device=self.loc.device)
        ids = torch.tensor(ids, device=self.loc.device).unsqueeze(1)

        new_state = self._replace(
                            ids=ids,
                            first_a=torch.tensor(first_a, device=self.loc.device).unsqueeze(1),
                            prev_a=prev_a,
                            visited_=torch.tensor(visited_, dtype=torch.uint8, device=self.loc.device),
                            lengths=torch.tensor(lengths, device=self.loc.device).unsqueeze(1),
                            cur_coord=torch.stack(cur_coord).unsqueeze(1),
                            i=torch.tensor(i, device=self.loc.device).unsqueeze(1))

        return new_state

