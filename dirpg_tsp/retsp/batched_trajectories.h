#ifndef BATCHED_TRAJECTORIES_H__
#define BATCHED_TRAJECTORIES_H__
#include <torch/extension.h>
#include <vector>

#include "batched_heaps.h"
using namespace std;


struct EmptyHeapsFilter{
    torch::Tensor non_empty_heaps; //boolean tensor for indexing the python state

    set<int> active_heaps;

    vector<int> ids_to_remove;
    int first_step;

    int budget_left;

    EmptyHeapsFilter(int batch_size, int search_budget){
        initialize(batch_size, search_budget);
    }
    void initialize(int batch_size, int search_budget){
        non_empty_heaps = torch::ones(batch_size, torch::kBool);
        //active_heaps.clear();
        for (int i=0; i<batch_size; i++)
            active_heaps.insert(i);
        first_step = batch_size;
        budget_left = search_budget;
    }

    void filter(int heap_id, int idx){
        if (first_step == 0){
            non_empty_heaps[idx] = false;
            ids_to_remove.push_back(heap_id);
        }
        else{
            // dont filter heaps after popping the root node
            first_step -= 1;
        }
    }

    void update(){
        for (int i=0; i<ids_to_remove.size(); i++){
            active_heaps.erase(ids_to_remove[i]);
            }

        non_empty_heaps = torch::ones(active_heaps.size(), torch::kBool);
        ids_to_remove.clear();
    }
    void dump(){
        py::print("---------- EmptyHeapsFilter ------------");
        py::print("active_heaps size: ", active_heaps.size());
        py::print("active_heaps: ", active_heaps);
        py::print("non_empty_heaps size: ", non_empty_heaps.size(0));
        py::print("non_empty_heaps: ", non_empty_heaps);

        py::print("first_step: ", first_step);
        py::print("ids_to_remove: ", ids_to_remove);

    }
};

//struct Trajectory;
struct OuterNode;
struct BatchedTrajectories{
    vector<float> costs;
    vector<float> objectives;
    torch::Tensor actions;

    vector<OuterNode> nodes_;
    BatchedTrajectories(int batch_size, int graph_size);
    void setTrajectory(int idx, float objective, OuterNode node);
    void dump(int idx);

};

struct ToptTdirect{
    BatchedTrajectories t_opt;
    BatchedTrajectories t_direct;
    vector<int> num_candidates;
    vector<int> prune_count;

    vector<bool> improvement_flag;
    vector<int> k_improvement_;

    float epsilon_;
    bool to_print_;


    ToptTdirect(int batch_size, int graph_size, float epsilon, int k_improvement, bool to_print);
    void setTrajectory(int idx, OuterNode node);   //int idx, Trajectory trajectory
    py::tuple get_t_opt_direct();

    void setEpsilon(float epsilon);
};

#endif