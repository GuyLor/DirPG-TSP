#ifndef BATCHED_TRAJECTORIES_H__
#define BATCHED_TRAJECTORIES_H__
#include <torch/extension.h>
#include <vector>

#include "batched_heaps.h"
using namespace std;

//struct Trajectory;
struct OuterNode;
struct BatchedTrajectories{
    vector<float> costs;
    vector<float> objectives;
    torch::Tensor actions;

    BatchedTrajectories(int batch_size, int graph_size);
    void setTrajectory(int idx, float objective, OuterNode node);

};

struct ToptTdirect{
    BatchedTrajectories t_opt;
    BatchedTrajectories t_direct;
    vector<int> num_candidates;

    ToptTdirect(int batch_size, int graph_size);
    void setTrajectory(int idx, float objective, OuterNode node);   //int idx, Trajectory trajectory
    py::tuple get_t_opt_direct();
};

#endif