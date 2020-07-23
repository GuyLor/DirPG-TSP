#ifndef A_STAR_SAMPLING_H__
#define A_STAR_SAMPLING_H__

#include <vector>

#include "node_allocator.h"
#include "batched_graphs.h"
#include "batched_heaps.h"
#include "batched_trajectories.h"

#include "mst_node.h"
#include "info_node.h"
#include "gumbel_state.h"


using namespace std;

struct EnvInfo{
    torch::Tensor batch_t;
    torch::Tensor batch_prev_city;
    torch::Tensor batch_next_actions;

    EnvInfo(int batch_size, int graph_size){
        batch_t = torch::empty(batch_size, torch::kInt64);
        batch_prev_city = torch::empty(batch_size, torch::kInt64);
        batch_next_actions = torch::empty({batch_size, graph_size}, torch::kBool);
    }
};

class AstarSampling {
 public:
  AstarSampling(int batch_size, int search_budget, int graph_size, float epsilon, float alpha, int seed);
  void initialize(torch::Tensor batch_start_city, torch::Tensor weights);
  void expand(torch::Tensor batch_special_action,
              torch::Tensor batch_logprobs,
              torch::Tensor batch_reward);

  EnvInfo popBatch();
  ToptTdirect getTrajectories();
  void clear();
  torch::Tensor non_empty_heaps;

 protected:

  int graph_size_;
  int batch_size_;
  float epsilon_;
  float alpha_;

  NodeAllocator<MstNode> mst_node_allocator_;
  NodeAllocator<GumbelState> gumbel_node_allocator_;
  NodeAllocator<InfoNode> info_node_allocator_;

  vector<OuterNode> current_nodes_;

  BatchedHeaps heaps_;
  BatchedGraphs graphs_;
  ToptTdirect trajectories_;

  default_random_engine random_generator_;

  float dynamicWeighting(int t);
  float computePriority(OuterNode outer_node, int idx);
  OuterNode split(OuterNode outer_node, int special_action, float cost, torch::Tensor logprobs);
};

#endif  // CPP_NODE_H__
