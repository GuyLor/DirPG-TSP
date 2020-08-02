#include "a_star_sampling.h"

#include <torch/extension.h>



AstarSampling::AstarSampling(int batch_size,
                             int search_budget,
                             int graph_size,
                             float epsilon,
                             float alpha,
                             bool dynamic_weighting,
                             int seed):
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
  dynamic_weighting_(dynamic_weighting),
  heaps_filter_(batch_size),
  random_generator_(seed)
  {}

OuterNode AstarSampling::split(OuterNode other_outer_node,
                               int special_action,
                               float cost,
                               torch::Tensor logprobs){

    MstNode *other_child_mst = other_outer_node.mst_node;
    MstNode *special_child_mst = mst_node_allocator_.split(other_child_mst,
                                                           special_action);
    GumbelState *other_child_gumbel = other_outer_node.gumbel_node;

    GumbelState *special_child_gumbel = gumbel_node_allocator_.split(other_child_gumbel,
                                                                     other_child_mst,
                                                                     special_action,
                                                                     logprobs);

    InfoNode *other_child_info = other_outer_node.info_node;
    InfoNode *special_child_info = info_node_allocator_.split(other_child_info,
                                                              special_action,
                                                              cost);
    other_outer_node.setNodes(other_child_mst, other_child_gumbel, other_child_info);


    OuterNode special_outer = OuterNode();
    special_outer.setNodes(special_child_mst, special_child_gumbel, special_child_info);
    return special_outer;
}

EnvInfo AstarSampling::popBatch(){


    int narrowed_batch_size = heaps_filter_.active_heaps.size();
    auto ids_it = heaps_filter_.active_heaps.begin();

    EnvInfo to_return(narrowed_batch_size, graph_size_);

    //py::print("before popBatch ");
    for (int sample_idx=0; sample_idx < narrowed_batch_size; sample_idx++){

        HeapNode heap_node(heaps_.pop(*ids_it, sample_idx, trajectories_, heaps_filter_));

        OuterNode node(heap_node.outer_node);

        to_return.batch_t[sample_idx] = node.info_node -> getT();

        to_return.batch_prev_city[sample_idx] = node.mst_node  -> getPrevCity();

        to_return.batch_next_actions[sample_idx] = node.mst_node -> getLegalNextActionMask();

        if(heaps_filter_.non_empty_heaps[sample_idx].item<bool>()) //first_step == 0 && !heaps_[*ids_it].is_empty())
            current_nodes_.push_back(node);

        ids_it++;



    }
    //py::print("after popBatch ");
    //py::print(current_nodes_.size());
    return to_return;
}

void AstarSampling::expand(torch::Tensor batch_special_action,
                           torch::Tensor batch_logprobs,
                           torch::Tensor batch_reward){

    //py::print("start");
    heaps_filter_.update();
    auto ids_it = heaps_filter_.active_heaps.begin();
    //py::print("current_nodes_");
    //py::print(current_nodes_.size());
    for (int idx=0; idx < current_nodes_.size(); idx++){

        int sample_idx = *ids_it;

        OuterNode special_outer = split(current_nodes_[idx],
                                        batch_special_action[idx].item<int>(),
                                        batch_reward[idx].item<float>(),
                                        batch_logprobs[idx]);
        special_outer.setCostAlphaMst(-graphs_.mstCost(sample_idx, *special_outer.mst_node), alpha_, dynamic_weighting_);
        heaps_.push(sample_idx, special_outer, epsilon_);
        if (!current_nodes_[idx].mst_node -> isAnyLegalAction()){
            current_nodes_[idx].setCostAlphaMst(-graphs_.mstCost(sample_idx, *current_nodes_[idx].mst_node),
                                                alpha_,
                                                dynamic_weighting_);
            heaps_.push(sample_idx, current_nodes_[idx], epsilon_);

        }
        ids_it++;
    }
    current_nodes_.clear();
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

        outer_node.setCostAlphaMst(-graphs_.mstCost(sample_idx, *outer_node.mst_node),
                                   alpha_,
                                   dynamic_weighting_);
        //heaps_.push(sample_idx, outer_node, computePriority(outer_node, sample_idx));
        heaps_.push(sample_idx, outer_node, epsilon_);

    }
}

ToptTdirect AstarSampling::getTrajectories(){
    return trajectories_;
}

torch::Tensor AstarSampling::getNonEmptyHeaps(){
    return heaps_filter_.non_empty_heaps;
}
void AstarSampling::clear(){
    heaps_.clearHeaps();
    heaps_filter_.initialize(batch_size_);
    trajectories_ = ToptTdirect(batch_size_, graph_size_);
    mst_node_allocator_.clear();
    gumbel_node_allocator_.clear();
    info_node_allocator_.clear();







}