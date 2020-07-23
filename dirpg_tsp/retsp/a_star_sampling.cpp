#include "a_star_sampling.h"

#include <torch/extension.h>



AstarSampling::AstarSampling(int batch_size, int search_budget, int graph_size, float epsilon, float alpha, int seed):
  mst_node_allocator_(batch_size * search_budget, graph_size),
  gumbel_node_allocator_(batch_size * search_budget, graph_size),
  info_node_allocator_(batch_size * search_budget, graph_size),

  heaps_(batch_size),
  graphs_(batch_size, graph_size),
  trajectories_(batch_size, graph_size),
  graph_size_(graph_size),
  batch_size_(batch_size),
  epsilon_(epsilon),
  alpha_(alpha),
  random_generator_(seed)
  {}

float AstarSampling::dynamicWeighting(int t){
    return 1.0 + alpha_ * (1.0 - (float)t / (float)graph_size_);
}

float AstarSampling::computePriority(OuterNode outer_node, int idx){
 /*
 if (idx == 0){
 py::print("-----------------------------------");
 py::print("T: ", outer_node.info_node -> getT());
 py::print("is t_opt: ", outer_node.info_node -> getIsTopt());
 py::print("max gumbel: ", outer_node.gumbel_node -> getMaxGumbel());
 py::print("cost: ", outer_node.info_node -> getCost());
 py::print("dynamic alpha: ", dynamicWeighting(outer_node.info_node -> getT()));
 py::print("mst cost: ", graphs_.mstCost(idx, *outer_node.mst_node));
 py::print("-----------------------------------");}
 */

 return outer_node.gumbel_node -> getMaxGumbel() +
        epsilon_*(outer_node.info_node -> getCost() +
        dynamicWeighting(outer_node.info_node -> getT()) * -graphs_.mstCost(idx, *outer_node.mst_node));
}



OuterNode AstarSampling::split(OuterNode outer_node,
                               int special_action,
                               float cost,
                               torch::Tensor logprobs){
    MstNode *other_child_mst = outer_node.mst_node;
    MstNode *special_child_mst = mst_node_allocator_.split(other_child_mst,
                                                           special_action);

    GumbelState *other_child_gumbel = outer_node.gumbel_node;

    GumbelState *special_child_gumbel = gumbel_node_allocator_.split(other_child_gumbel,
                                                                     other_child_mst,
                                                                     special_action,
                                                                     logprobs);


    InfoNode *other_child_info = outer_node.info_node;
    InfoNode *special_child_info = info_node_allocator_.split(other_child_info,
                                                              special_action,
                                                              cost);

    outer_node.setNodes(other_child_mst, other_child_gumbel, other_child_info);

    OuterNode special_outer = OuterNode();
    special_outer.setNodes(special_child_mst, special_child_gumbel, special_child_info);
    return special_outer;
}

EnvInfo AstarSampling::popBatch(){
    EnvInfo to_return(batch_size_, graph_size_);
    for (int sample_idx=0; sample_idx < batch_size_; sample_idx++){
        HeapNode heap_node(heaps_.pop(sample_idx, trajectories_, non_empty_heaps));
        OuterNode node(heap_node.outer_node);
        to_return.batch_t[sample_idx] = node.info_node -> getT();

        to_return.batch_prev_city[sample_idx] = node.mst_node  -> getPrevCity();

        to_return.batch_next_actions[sample_idx] = node.mst_node -> getLegalNextActionMask();

        current_nodes_.push_back(node);
    }

    return to_return;
}

void AstarSampling::expand(torch::Tensor batch_special_action,
                           torch::Tensor batch_logprobs,
                           torch::Tensor batch_reward){

    for (int sample_idx=0; sample_idx < batch_size_; sample_idx++){
        if (!non_empty_heaps[sample_idx].item<bool>()) continue;
        //py::print("-----------------   ",sample_idx,   "   -----------------");

        OuterNode special_outer = split(current_nodes_[sample_idx],
                                        batch_special_action[sample_idx].item<int>(),
                                        batch_reward[sample_idx].item<float>(),
                                        batch_logprobs[sample_idx]);

        heaps_.push(sample_idx, special_outer, computePriority(special_outer, sample_idx)); // batch_done[sample_idx].item<bool>()

        if (!current_nodes_[sample_idx].mst_node -> isAnyLegalAction()){
            heaps_.push(sample_idx, current_nodes_[sample_idx], computePriority(current_nodes_[sample_idx], sample_idx)); //, batch_done[sample_idx].item<bool>()
        }
        //if (sample_idx == 0)
        //   heaps_[sample_idx].printHeap();

        current_nodes_.clear();
    }
}

void AstarSampling::initialize(torch::Tensor batch_start_city, torch::Tensor weights){
    //mst_node_allocator_.clear();
    //gumbel_node_allocator_.clear();
    //info_node_allocator_.clear();
    for (int sample_idx=0; sample_idx < batch_size_; sample_idx++){
        MstNode *mst_node = mst_node_allocator_.getNew(); // root
        GumbelState *gumbel_node = gumbel_node_allocator_.getNew(); // root
        InfoNode *info_node = info_node_allocator_.getNew(); // root

        int start_city = batch_start_city[sample_idx].item<int>();

        non_empty_heaps = at::ones(batch_size_ ,torch::kUInt8 );

        mst_node -> setRoot(start_city);
        gumbel_node -> setRoot(random_generator_);
        info_node -> setRoot(start_city);
        //info_node -> setLastStepCost(last_potential_cost);


        OuterNode outer_node = OuterNode();
        outer_node.setNodes(mst_node, gumbel_node, info_node);
        // fill graphs
        float dm[graph_size_*graph_size_];
        for(int idx=0; idx < graph_size_*graph_size_; idx++){
            dm[idx] = weights[sample_idx][idx].item<float>();
        }
        graphs_.setWeights(sample_idx, dm);


        heaps_.push(sample_idx, outer_node, computePriority(outer_node, sample_idx));

    }
}

ToptTdirect AstarSampling::getTrajectories(){
    return trajectories_;
}
void AstarSampling::clear(){
    heaps_.clearHeaps();
    trajectories_ = ToptTdirect(batch_size_, graph_size_);
    mst_node_allocator_.clear();
    gumbel_node_allocator_.clear();
    info_node_allocator_.clear();







}