#include "batched_trajectories.h"

//EmptyHeapsFilter::EmptyHeapsFilter





BatchedTrajectories::BatchedTrajectories(int batch_size, int graph_size){
    costs.resize(batch_size, 1e8);
    objectives.resize(batch_size, -1e8);
    actions = torch::empty({batch_size, graph_size}, torch::kInt32);
    nodes_.resize(batch_size);
}

void BatchedTrajectories::setTrajectory(int idx, float objective, OuterNode node){
    costs[idx] = node.info_node -> getCost(); //trajectory.cost;
    objectives[idx] = objective;
    nodes_[idx] = node;
    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    actions[idx] = torch::from_blob((node.info_node -> getActions()).data(), actions.size(1), opts); //trajectory.actions;
    }

void BatchedTrajectories::dump(int idx){
    py::print("cost: ",costs[idx]);
    py::print("objective: ",objectives[idx]);
    py::print("actions: ",actions[idx]);
    nodes_[idx].dump();

}

ToptTdirect::ToptTdirect(int batch_size, int graph_size, float epsilon, int k_improvement, bool to_print):
    t_opt(batch_size,graph_size),
    t_direct(batch_size, graph_size),
    epsilon_(epsilon),
    to_print_(to_print)
    {

        num_candidates.resize(batch_size);
        prune_count.resize(batch_size);
        improvement_flag.resize(batch_size, false);
        k_improvement_.resize(batch_size, k_improvement);

    }

void ToptTdirect::setTrajectory(int idx, OuterNode node){
    num_candidates[idx] += 1;
    float objective = node.computeObjective(epsilon_);
    //py::print("idx: ", idx);
    //node.dump();
    if(idx == 0 && to_print_){
        py::print("new trajectory: ",num_candidates[idx]);
        node.dump();}
    if (num_candidates[idx] == 1){

        //py::print("objective: ", objective);
        t_opt.setTrajectory(idx, objective, node);
        t_direct.setTrajectory(idx, objective, node);
    }
    else if (objective > t_direct.objectives[idx]){
        //else if (node.info_node -> getCost() < t_direct.costs[idx]){
        //py::print("objective: ", objective, " >  best direct: ",t_direct.objectives[idx] );
        t_direct.setTrajectory(idx, objective, node);
        k_improvement_[idx] -= 1;
        if (k_improvement_[idx] == 0)
            improvement_flag[idx] = true;
    }
}

py::tuple ToptTdirect::get_t_opt_direct(){
    return py::make_tuple(t_opt, t_direct);
}

void ToptTdirect::setEpsilon(float epsilon){
    epsilon_ = epsilon;
}
