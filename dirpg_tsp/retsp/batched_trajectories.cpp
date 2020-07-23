#include "batched_trajectories.h"


BatchedTrajectories::BatchedTrajectories(int batch_size, int graph_size){
    costs.resize(batch_size);
    objectives.resize(batch_size);
    actions = torch::empty({batch_size, graph_size}, torch::kInt32); //
}

void BatchedTrajectories::setTrajectory(int idx, float objective, OuterNode node){
    costs[idx] = node.info_node -> getCost(); //trajectory.cost;
    objectives[idx] = objective;

    auto opts = torch::TensorOptions().dtype(torch::kInt32);
    actions[idx] = torch::from_blob((node.info_node -> getActions()).data(), actions.size(1), opts); //trajectory.actions;
    }


ToptTdirect::ToptTdirect(int batch_size, int graph_size):
    t_opt(batch_size,graph_size),
    t_direct(batch_size, graph_size){

        num_candidates.resize(batch_size);
}

void ToptTdirect::setTrajectory(int idx, float objective, OuterNode node){
    num_candidates[idx] += 1;
    if (num_candidates[idx] == 1){
        t_opt.setTrajectory(idx, objective, node);
        t_direct.setTrajectory(idx, objective, node);
    }
    else if (objective > t_opt.objectives[idx]){
        t_direct.setTrajectory(idx, objective, node);
    }
}

py::tuple ToptTdirect::get_t_opt_direct(){
    return py::make_tuple(t_opt, t_direct);
}
