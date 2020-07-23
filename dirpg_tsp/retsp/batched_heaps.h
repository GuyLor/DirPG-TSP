#ifndef BATCHED_HEAPS_H__
#define BATCHED_HEAPS_H__
#include <torch/extension.h>
#include <vector>
#include "batched_trajectories.h"
#include "mst_node.h"
#include "info_node.h"
#include "gumbel_state.h"


using namespace std;
struct ToptTdirect;
struct OuterNode{

    MstNode *mst_node;   // attrs: visited_mask_, legal_next_action_mask_, first_node_, last_node_
    GumbelState *gumbel_node;  // attrs: max_gumbel_, logprob_so_far_
    InfoNode *info_node;

    OuterNode(){
    }

    void copyState(const OuterNode *other_node) {
      gumbel_node = other_node->gumbel_node;
      mst_node = other_node->mst_node;
      info_node = other_node->info_node;
    }
    void setNodes(MstNode *mst_node_, GumbelState *gumbel_node_, InfoNode *info_node_){

        mst_node = mst_node_;
        gumbel_node = gumbel_node_;
        info_node = info_node_;
    }
    void dump(){
        mst_node -> dump();
        gumbel_node -> dump();
        info_node -> dump();
    }
};


struct HeapNode {
    float priority;
    bool t_opt;
    bool done;
    //bool is_any_next_legals;
    OuterNode outer_node;

    HeapNode(){}
    ~HeapNode(){}
    HeapNode(float priority_, OuterNode outer_node_)
    :
    priority(priority_),
    t_opt(outer_node_.info_node -> getIsTopt()),
    done(outer_node_.info_node -> getIsDone()),
    //is_any_next_legals(outer_node_.mst_node -> isAnyLegalAction()),
    outer_node(outer_node_)
    {}
    HeapNode(const HeapNode &other){
        priority = other.priority;
        t_opt = other.t_opt;
        done = other.done;
        //is_any_next_legals = other.is_any_next_legals;
        outer_node = other.outer_node;
    }

    bool operator<(const HeapNode &other) const {
         if (t_opt == other.t_opt){
            return priority < other.priority;
         }
         if (t_opt && !other.t_opt){
            return false;
         }
         else{
            return true;
         }
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
        BatchedHeaps(int batch_size);

        HeapNode pop(int sample_idx, ToptTdirect &trajectories, torch::Tensor &non_empty_heaps);
        void push(int sample_idx, OuterNode outer_node, float priority);
        void clearHeaps();
        void printHeap(int sample_idx);

        Heap operator[](int i) const{
            return heaps_[i];
        }


    protected:
        int batch_size_;
        vector<Heap> heaps_;

   };

#endif