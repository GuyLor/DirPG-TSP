#ifndef BATCHED_HEAPS_H__
#define BATCHED_HEAPS_H__
#include <torch/extension.h>
#include <vector>
#include <math.h>

#include "batched_trajectories.h"
#include "mst_node.h"
#include "info_node.h"
#include "gumbel_state.h"


using namespace std;
struct ToptTdirect;
struct EmptyHeapsFilter;
struct OuterNode{

    MstNode *mst_node;   // attrs: visited_mask_, legal_next_action_mask_, first_node_, last_node_
    GumbelState *gumbel_node;  // attrs: max_gumbel_, logprob_so_far_
    InfoNode *info_node;

    float priority;
    float upper_bound;
    float mst_val;
    int steps_left;

    OuterNode(){
    }

    void copyState(const OuterNode *other_node) {
      gumbel_node = other_node->gumbel_node;
      mst_node = other_node->mst_node;
      info_node = other_node->info_node;
      priority = other_node->priority;
      upper_bound = other_node->upper_bound;
      mst_val = other_node->mst_val;
      steps_left = other_node->steps_left;

    }

    void setNodes(MstNode *mst_node_, GumbelState *gumbel_node_, InfoNode *info_node_){
        mst_node = mst_node_;
        gumbel_node = gumbel_node_;
        info_node = info_node_;
    }

    void setPriorityAndUpperBound(float mst, float epsilon, float alpha, bool dynamic_weighting, int priority_function){

        float gumbel = gumbel_node -> getMaxGumbel();
        float cost_so_far = info_node -> getCost();

        upper_bound = gumbel + epsilon * (- cost_so_far - mst);
        mst_val = mst;
        //upper_bound = -cost_so_far - alpha * mst;
        steps_left = info_node -> getGraphSize() - info_node -> getT();
        if (dynamic_weighting){
            alpha =  1.0 +alpha * (1 - (float)(info_node -> getT()) / (float)(info_node -> getGraphSize()));

            }
        if (priority_function == 0) //gumbel_top_k
            priority = gumbel;
        else
            priority = gumbel + epsilon * (-cost_so_far - alpha * mst );
        /*
        if (info_node -> getIsDone()){
        py::print("--------");
        py::print("gumbel: ",gumbel);
        py::print("cost_so_far: ",cost_so_far);
        py::print("mst: ",mst);
        py::print("epsilon*(cost_so_far + alpha * mst): ",epsilon*(-cost_so_far - alpha * mst));
        //priority = (epsilon+gumbel )/(cost_so_far + alpha * mst);
        py::print("priority: ",priority);
        }
        */

    }

    //float computePriority(float epsilon){
    //   return gumbel_node -> getMaxGumbel() + epsilon * (10 - cost_alpha_mst);
    //}

    float computeObjective(float epsilon){
       //return -info_node -> getCost();
       return gumbel_node -> getMaxGumbel() + epsilon * -info_node -> getCost();
    }
    void dump(){
        py::print("*** OuterNode ***");
        mst_node -> dump();
        gumbel_node -> dump();
        info_node -> dump();

        py::print("priority: ", priority);
        py::print("upper bound: ", upper_bound);
        py::print("mst: ", mst_val);
        py::print("****************");
    }
};


struct HeapNode {
    float priority;
    bool t_opt;
    bool dfs_like;
    bool done;
    //bool is_any_next_legals;
    OuterNode outer_node;

    HeapNode(){}
    ~HeapNode(){}
    HeapNode(OuterNode outer_node_)
    :
    priority(outer_node_.priority),
    t_opt(outer_node_.info_node -> getIsTopt()),
    dfs_like(outer_node_.info_node -> getDfs()),
    done(outer_node_.info_node -> getIsDone()),
    outer_node(outer_node_)
    {
    }

    HeapNode(const HeapNode &other){
        priority = other.priority;
        t_opt = other.t_opt;
        dfs_like = other.dfs_like;
        done = other.done;
        outer_node = other.outer_node;
    }

    bool to_prune(float lower_bound){
        //if (outer_node.upper_bound < lower_bound){
        //    py::print("outer_node.upper_bound ", outer_node.upper_bound);
        //    outer_node.dump();
        //    py::print("lower_bound ", lower_bound);}

       return outer_node.upper_bound < lower_bound;
    }

    bool operator<(const HeapNode &other) const {
         if (done) return false;
         if ((t_opt && !other.t_opt )|| dfs_like) return false;
         if (t_opt == other.t_opt) return priority < other.priority;

         return true;

    }
    void dump(){
      py::print("priority:  ", priority);
      py::print("t_opt:  ", t_opt);
      py::print("done:  ", done);
      outer_node.dump();
    }
};

class Heap {
    public:
        Heap(){};
        void push(HeapNode new_node);

        HeapNode pop();

        bool is_empty();

        void clearHeap();

        void printHeap();

    protected:
        std::vector<HeapNode> elements_;
};

class BatchedHeaps{
    public:
        BatchedHeaps(int batch_size, bool prune_flag);

        HeapNode pop(int heap_id, int sample_idx, ToptTdirect &trajectories, EmptyHeapsFilter &heaps_filter);
        void push(int heap_id, OuterNode outer_node);
        void clearHeaps();
        void printHeap(int heap_id);

        Heap operator[](int i) const{
            return heaps_[i];
        }

    protected:
        int batch_size_;
        vector<Heap> heaps_;
        bool prune_flag_;
   };

#endif