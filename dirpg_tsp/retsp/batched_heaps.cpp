
#include "batched_heaps.h"
#include<vector>
using namespace std;


void Heap::push(HeapNode new_node) {
     elements_.push_back(new_node);
     push_heap(elements_.begin(), elements_.end());
}

HeapNode Heap::pop(){
    HeapNode result = elements_.front();
    pop_heap(elements_.begin(), elements_.end());

    elements_.pop_back();
    return result;
}

bool Heap::is_empty() {
    return elements_.empty();
};

void Heap::clearHeap(){
    elements_.clear();
}

void Heap::printHeap(){
   py::print("#########",  elements_.size()   ,"##########");
   for (int i=0; i<elements_.size(); i++){
    py::print("------------------------");
    elements_[i].dump();
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////
BatchedHeaps::BatchedHeaps(int batch_size){

    batch_size_ = batch_size;
    for (int i=0; i<batch_size_; i++){
        heaps_.push_back(Heap());
    }
}

HeapNode BatchedHeaps::pop(int sample_idx, ToptTdirect &trajectories, torch::Tensor &non_empty_heaps){
        bool to_pop = true; //!heaps_[sample_idx].is_empty();
        HeapNode node;
        while (to_pop ){
            node = heaps_[sample_idx].pop();
            bool non_empty_heap = !heaps_[sample_idx].is_empty();
            //to_pop = (node.done || node.is_any_next_legals) && full_heap;
            to_pop = node.done  && non_empty_heap;
            non_empty_heaps[sample_idx] = non_empty_heap;

            if(node.done){
                trajectories.setTrajectory(sample_idx, node.priority, node.outer_node);
            }
        }
        return node;
    }

void BatchedHeaps::push(int sample_idx, OuterNode outer_node, float priority){
        HeapNode node(priority, outer_node);
        //node.dump();
        heaps_[sample_idx].push(node);
    }

void BatchedHeaps::clearHeaps(){
    for (int i=0; i<batch_size_; i++){
        heaps_[i].clearHeap();
    }
}

void BatchedHeaps::printHeap(int sample_idx){
    heaps_[sample_idx].printHeap();
}