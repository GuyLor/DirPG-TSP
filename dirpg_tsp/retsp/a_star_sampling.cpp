#include "a_star_sampling.h"

#include <torch/extension.h>




AstarSampling::AstarSampling(int batch_size,
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
                             int seed):
  mst_node_allocator_(batch_size * search_budget + batch_size, graph_size),
  gumbel_node_allocator_(batch_size * search_budget + batch_size, graph_size),
  info_node_allocator_(batch_size * search_budget + batch_size, graph_size),

  heaps_(batch_size, !prune_flag),
  graphs_(batch_size, graph_size),
  graphs_tsp_(batch_size, graph_size),
  trajectories_(batch_size, graph_size, epsilon, k_improvement, false),
  graph_size_(graph_size),
  batch_size_(batch_size),
  search_budget_(search_budget),
  epsilon_(epsilon),
  alpha_(alpha),
  dfs_like_(dfs_like),
  optimal_heuristic_(optimal_heuristic),
  dynamic_weighting_(dynamic_weighting),
  priority_function_(priority_function),
  k_improvement_(k_improvement),
  heaps_filter_(batch_size, search_budget),
  to_print_(false),
  random_generator_(seed)
  {}

OuterNode AstarSampling::split(OuterNode other_outer_node,
                               int special_action,
                               float cost,
                               vector<float> logprobs){

    clock_t before_split_mst, before_split_gumbel, before_split_info, after_split_info;
    before_split_mst = clock();

    MstNode *other_child_mst = other_outer_node.mst_node;
    MstNode *special_child_mst = mst_node_allocator_.split(other_child_mst,
                                                           special_action);

    before_split_gumbel = clock();
    split_mst_time += double (before_split_gumbel - before_split_mst);

    GumbelState *other_child_gumbel = other_outer_node.gumbel_node;

    GumbelState *special_child_gumbel = gumbel_node_allocator_.split(other_child_gumbel,
                                                                     other_child_mst,
                                                                     special_action,
                                                                     logprobs);

    before_split_info = clock();
    split_gumbel_time += double (before_split_info - before_split_gumbel);

    InfoNode *other_child_info = other_outer_node.info_node;
    InfoNode *special_child_info = info_node_allocator_.split(other_child_info,
                                                              special_action,
                                                              cost,
                                                              dfs_like_);
    after_split_info = clock();
    split_info_time += double (after_split_info - before_split_info);

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
        if (sample_idx ==0 && to_print_)
            heap_node.dump();
        OuterNode node(heap_node.outer_node);


        to_return.batch_t[sample_idx] = node.info_node -> getT();

        to_return.batch_prev_city[sample_idx] = node.mst_node  -> getPrevCity();

        to_return.batch_next_actions[sample_idx] = node.mst_node -> getLegalNextActionMaskTorch();

        if(heaps_filter_.non_empty_heaps[sample_idx].item<bool>())
            current_nodes_.push_back(node);

        ids_it++;



    }
    //py::print("after popBatch ");
    //py::print(current_nodes_.size());
    return to_return;
}

void AstarSampling::expand(vector<int> batch_special_action,
                           vector<vector<float>> batch_logprobs,
                           vector<float> batch_reward){

    //py::print("start");

    auto ids_it = heaps_filter_.active_heaps.begin();
    heaps_filter_.budget_left --;
    //py::print("current_nodes_");
    //py::print(current_nodes_.size());



    for (int idx=0; idx < current_nodes_.size(); idx++){

        int sample_idx = *ids_it;
        clock_t before_split, before_mst_sp, before_push_sp,after_push_sp, before_mst_oth, before_push_oth, after_push_oth;
        before_split = clock();

        OuterNode special_outer = split(current_nodes_[idx],
                                        batch_special_action[idx],
                                        batch_reward[idx],
                                        batch_logprobs[idx]);

        before_mst_sp = clock();
        split_time += double (before_mst_sp - before_split);

        special_outer.setPriorityAndUpperBound(computeHeuristic_(sample_idx, special_outer),
                                               epsilon_,
                                               alpha_,
                                               dynamic_weighting_,
                                               priority_function_);
        before_push_sp = clock();
        mst_special_time += double (before_push_sp - before_mst_sp);

        heaps_.push(sample_idx, special_outer);

        after_push_sp = clock();

        mst_special_time += double (after_push_sp - before_push_sp);

        if (!current_nodes_[idx].mst_node -> isAnyLegalAction()){
            before_mst_oth = clock();
            //if (sample_idx == 0)
            //    py::print("before push other");
            current_nodes_[idx].setPriorityAndUpperBound(computeHeuristic_(sample_idx, current_nodes_[idx]),
                                                         epsilon_,
                                                         alpha_,
                                                         dynamic_weighting_,
                                                         priority_function_);
            before_push_oth = clock();
            mst_other_time += double (before_push_oth - before_mst_oth);

            heaps_.push(sample_idx, current_nodes_[idx]);
            after_push_oth = clock();

            push_other_time += double (after_push_oth - before_push_oth);

        }
        ids_it++;
    }
    //heaps_[0].printHeap();
    current_nodes_.clear();

}

int seed = 2;
default_random_engine engine(seed);

void AstarSampling::initialize(vector<int> batch_start_city, vector<vector<float>> weights){
    //mst_node_allocator_.clear();
    //gumbel_node_allocator_.clear();
    //info_node_allocator_.clear();
    split_time=0.0; mst_special_time=0.0; mst_other_time=0.0; push_special_time=0.0; push_other_time=0.0;
    split_mst_time=0.0; split_gumbel_time=0.0; split_info_time=0.0;
    for (int sample_idx=0; sample_idx < batch_size_; sample_idx++){
        MstNode *mst_node = mst_node_allocator_.getNew(); // root
        GumbelState *gumbel_node = gumbel_node_allocator_.getNew(); // root
        InfoNode *info_node = info_node_allocator_.getNew(); // root

        int start_city = batch_start_city[sample_idx];

        engine.seed(seed*(sample_idx+1));
        mst_node -> setRoot(start_city);
        gumbel_node -> setRoot(engine);
        info_node -> setRoot(start_city);
        //info_node -> setLastStepCost(last_potential_cost);
        OuterNode outer_node = OuterNode();
        outer_node.setNodes(mst_node, gumbel_node, info_node);
        // fill graphs

        float dm[graph_size_*graph_size_];
        for(int idx=0; idx < graph_size_*graph_size_; idx++){
            dm[idx] = weights[sample_idx][idx];
        }
        if (!optimal_heuristic_){
            graphs_.setWeights(sample_idx, dm);
            }
        else{
            graphs_tsp_.setWeights(sample_idx, dm);
            //py::print("tsp first: ", computeHeuristic_(sample_idx, outer_node));
            }

        outer_node.setPriorityAndUpperBound(computeHeuristic_(sample_idx, outer_node),
                                            epsilon_,
                                            alpha_,
                                            dynamic_weighting_,
                                            priority_function_);
        //heaps_.push(sample_idx, outer_node, computePriority(outer_node, sample_idx));
        heaps_.push(sample_idx, outer_node);

    }
    seed++;
}
float AstarSampling::computeHeuristic_(int sample_idx, OuterNode node){
    if (optimal_heuristic_){
        return graphs_tsp_.tspCost(sample_idx, *node.mst_node);
    }

    else{
        return graphs_.mstCost(sample_idx, *node.mst_node);
        }

}

float AstarSampling::computeTSP(int sample_idx, MstNode node){
        return graphs_tsp_.tspCost(sample_idx, node);


}
void AstarSampling::printTime(){
    double sum_total = split_time + mst_special_time + mst_other_time + push_special_time + push_other_time;
    double sum_splits = split_mst_time + split_gumbel_time + split_info_time;
    py::print("----------- relative TIME -----------");
    py::print("total split: ", split_time/sum_total );
    py::print("####");

    py::print("split_mst: ", split_mst_time/sum_splits);
    py::print("split_gumbel: ", split_gumbel_time/sum_splits);
    py::print("split_info: ", split_info_time/sum_splits);
    py::print("####");

    py::print("mst_special: ", mst_special_time/sum_total );
    py::print("mst_other: ", mst_other_time/sum_total );
    py::print("push_special: ", push_special_time/sum_total );
    py::print("push_other: ", push_other_time/sum_total );
}

ToptTdirect AstarSampling::getTrajectories(){
    return trajectories_;
}

torch::Tensor AstarSampling::getNonEmptyHeaps(){
    return heaps_filter_.non_empty_heaps;
}

int AstarSampling::getEmptyHeapsCount(){
    return batch_size_ - heaps_filter_.active_heaps.size();
}

void AstarSampling::setEpsilonAlpha(float epsilon, float alpha){
    alpha_ = alpha;
    epsilon_ = epsilon;
}

void AstarSampling::setKImprovement(int k){
    k_improvement_ = k;
}


py::tuple AstarSampling::getEpsilonAlpha(){
    return py::make_tuple(epsilon_, alpha_);
}

void AstarSampling::updateHeapFilter(){
    heaps_filter_.update();
}

void AstarSampling::setToPrint(bool t){
    to_print_ = t;
}

void AstarSampling::clear(){
    heaps_.clearHeaps();
    heaps_filter_ = EmptyHeapsFilter(batch_size_, search_budget_);
    trajectories_ = ToptTdirect(batch_size_, graph_size_, epsilon_, k_improvement_, to_print_);
    mst_node_allocator_.clear();
    gumbel_node_allocator_.clear();
    info_node_allocator_.clear();



}