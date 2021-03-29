#ifndef A_STAR_SAMPLING_H__
#define A_STAR_SAMPLING_H__

#include <vector>
#include <set>
#include <ctime> // time_t
#include <cstdio>



#include "node_allocator.h"
#include "batched_graphs.h"
#include "batched_graphs_tsp.h"
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

        //py::print("required grad:  ",batch_next_actions.requires_grad() );
    }
};

class AstarSampling {
 public:
  AstarSampling(int batch_size,
                int search_budget,
                int graph_size,
                float epsilon,
                float alpha,
                bool dynamic_weighting,
                int priority_function,
                bool prune_flag,
                bool dfs_like,
                bool optimal_heuristic,
                int k_improvement,
                int seed);
  void initialize(vector<int> batch_start_city, vector<vector<float>> weights);
  void expand(vector<int> batch_special_action,
              vector<vector<float>> batch_logprobs,
              vector<float> batch_reward);

  EnvInfo popBatch();
  ToptTdirect getTrajectories();
  void clear();
  torch::Tensor getNonEmptyHeaps();
  int getEmptyHeapsCount();
  void printTime();
  void updateHeapFilter();
  void setEpsilonAlpha(float epsilon, float alpha);
  void setKImprovement(int k);
  py::tuple getEpsilonAlpha();
  void setToPrint(bool t);
  float computeTSP(int sample_idx, MstNode node);


 protected:

  int graph_size_;
  int batch_size_;
  int search_budget_;
  float epsilon_;
  float alpha_;
  bool dynamic_weighting_;
  bool dfs_like_;
  bool optimal_heuristic_;
  int priority_function_;
  int k_improvement_;

  bool to_print_;

  double split_time, mst_special_time, mst_other_time, push_special_time, push_other_time;
  double split_mst_time, split_gumbel_time, split_info_time;

  NodeAllocator<MstNode> mst_node_allocator_;
  NodeAllocator<GumbelState> gumbel_node_allocator_;
  NodeAllocator<InfoNode> info_node_allocator_;

  vector<OuterNode> current_nodes_;

  BatchedHeaps heaps_;
  BatchedGraphs graphs_;
  BatchedGraphsOptimal graphs_tsp_;
  ToptTdirect trajectories_;
  EmptyHeapsFilter heaps_filter_;

  default_random_engine random_generator_;


  float dynamicWeighting(int t);
  float computePriority(OuterNode outer_node, int idx);
  float computeHeuristic_(int sample_idx, OuterNode node);
  OuterNode split(OuterNode outer_node, int special_action, float cost, vector<float> logprobs);

};

#endif  // CPP_NODE_H__
