
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

HeapNode BatchedHeaps::pop(int heap_id, int sample_idx, ToptTdirect &trajectories, EmptyHeapsFilter &heaps_filter){
        bool to_pop = true; //!heaps_[sample_idx].is_empty();
        HeapNode node;
        while (to_pop){
            node = heaps_[heap_id].pop();
            bool empty_heap = heaps_[heap_id].is_empty();
            //if (!node.t_opt){
            //    trajectories.t_direct.objectives[sample_idx]
            //}
            bool to_prune = node.to_prune(trajectories.t_direct.costs[heap_id]);
            if (to_prune) trajectories.prune_count[heap_id] += 1;
            to_pop = (node.done || to_prune)  && !empty_heap;

            if (empty_heap) {heaps_filter.filter(heap_id, sample_idx);}
            //non_empty_heaps[sample_idx] = non_empty_heap;

            if(node.done)
                trajectories.setTrajectory(heap_id, node.priority, node.outer_node);
        }
        return node;
    }

void BatchedHeaps::push(int heap_id, OuterNode outer_node, float epsilon){

    HeapNode node(outer_node.computePriority(epsilon), outer_node);
    //node.dump();
    heaps_[heap_id].push(node);
}

void BatchedHeaps::clearHeaps(){
    for (int i=0; i<batch_size_; i++){
        heaps_[i].clearHeap();
    }
}

void BatchedHeaps::printHeap(int heap_id){
    heaps_[heap_id].printHeap();
}