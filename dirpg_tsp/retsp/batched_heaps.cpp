
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
BatchedHeaps::BatchedHeaps(int batch_size, bool prune_flag){

    batch_size_ = batch_size;
    prune_flag_ = prune_flag;
    for (int i=0; i<batch_size_; i++){
        heaps_.push_back(Heap());
    }
}

HeapNode BatchedHeaps::pop(int heap_id, int sample_idx, ToptTdirect &trajectories, EmptyHeapsFilter &heaps_filter){
        bool to_pop = true; //!heaps_[sample_idx].is_empty();
        bool stop_first_improvement = false;
        HeapNode node;
        while (to_pop){
            node = heaps_[heap_id].pop();
            bool empty_heap = heaps_[heap_id].is_empty();
            //if (!node.t_opt){
            //    trajectories.t_direct.objectives[sample_idx]
            //}
            bool to_prune =  (prune_flag_ && node.to_prune(trajectories.t_direct.objectives[heap_id])) ||
                             (node.outer_node.steps_left > heaps_filter.budget_left);
            if (to_prune){
               trajectories.prune_count[heap_id] += 1;
               /*
               if(heap_id == 0){
               py::print("-------trajectory--------");
               trajectories.t_direct.dump(heap_id);
               py::print("--------pruned node-------");
               node.dump();
               }
               */

               }


            if(node.done){
                trajectories.setTrajectory(heap_id, node.outer_node);
                if (trajectories.improvement_flag[heap_id])
                    stop_first_improvement = true;
                }


            if (empty_heap || stop_first_improvement) {heaps_filter.filter(heap_id, sample_idx);}
            //non_empty_heaps[sample_idx] = non_empty_heap;
            to_pop = (node.done || to_prune)  && !empty_heap && !stop_first_improvement;


        }
        return node;
    }

void BatchedHeaps::push(int heap_id, OuterNode outer_node){

    HeapNode node(outer_node);
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