
/* Allocating and de-allocating memory for nodes is expensive. When we're
 * performing search with a fixed budget of environment interactions, we
 * know exactly how many nodes we need to expand, and the amount of memory
 * needed is constant across all searches. So let's allocate that memory
 * once and re-use it across the entire execution of training.
 *
 * This class manages CppTspNode memory to achieve this.
 */
#ifndef NODE_ALLOCATOR_H__
#define NODE_ALLOCATOR_H__


#include <vector>
//#include "node_allocator.h"
#include <torch/extension.h>
#include "assert.h"

#include "mst_node.h"
#include "gumbel_state.h"
#include "info_node.h"



using namespace std;

template<typename T>
class NodeAllocator {

 public:
  NodeAllocator<T>(int max_num_nodes, int graph_size);
  //NodeAllocator<T>();
  ~NodeAllocator<T>();

  // Get a "new" node. It doesn't actually allocate memory (it just re-uses)
  // but otherwise acts like `new CppTspNode(graph_size)`.
  T *getNew();

  // Reset all "allocations".
  void clear();

  // Splits `parent` into two nodes. One of the new nodes with overwrite the
  // memory of `parent`. The other will be returned as a new pointer (that the
  // allocator will own the memory of).
  //T *split(T *parent, int special_child);

  T *split(MstNode *parent, int special_child);
  T *split(GumbelState *parent_gumbel, MstNode *parent_mst, int special_action, torch::Tensor logprobs);
  T *split(InfoNode *parent, int special_child, float cost);

 protected:
  vector<T *> nodes_;
  vector<bool> is_allocated_;
  int next_to_allocate_;
  int max_num_nodes_;
};

template<typename T>
NodeAllocator<T>::NodeAllocator(int max_num_nodes, int graph_size) {
  nodes_.resize(max_num_nodes);
  is_allocated_.resize(max_num_nodes);
  max_num_nodes_ = max_num_nodes;
  next_to_allocate_ = 0;

  for (int i = 0; i < max_num_nodes; i++) {
    nodes_[i] = new T(graph_size);
  }
}


template<typename T>
NodeAllocator<T>::~NodeAllocator() {
  for (int i = 0; i < max_num_nodes_; i++) {
    delete nodes_[i];
  }
 }



template<typename T>
T *NodeAllocator<T>::getNew() {
  assert(next_to_allocate_ < max_num_nodes_);
  is_allocated_[next_to_allocate_] = true;
  return nodes_[next_to_allocate_++];
}


template<typename T>
void NodeAllocator<T>::clear() {
  /* Set state so we can allocate anything, but don't bother clearing nodes_.
   *
   * If there are dangling pointers to nodes that have been previously allocated
   * then the memory will stay in tact, but it might get overridden with new
   * nodes.
   */
  fill(is_allocated_.begin(), is_allocated_.end(), false);
  next_to_allocate_ = 0;
}


template<typename T>
T *NodeAllocator<T>::split(MstNode *parent, int special_child) {
  // Modifies `parent` in place. Assumes it's not needed any more.
  MstNode *other_children_node = parent;

  MstNode *special_child_node = getNew();
  special_child_node->CopyState(parent);

  special_child_node->transformToSpecialChild(special_child);
  other_children_node->transformToOtherChildren(special_child);

  return special_child_node;
}

template<typename T>
T *NodeAllocator<T>::split(GumbelState *parent_gumbel, MstNode *parent_mst, int special_action, torch::Tensor logprobs) {
  // Modifies `parent` in place. Assumes it's not needed any more.
  GumbelState *other_children_node = parent_gumbel;

  GumbelState *special_child_node = getNew();
  special_child_node->CopyState(parent_gumbel);

  special_child_node->transformToSpecialChild(special_action, logprobs);
  other_children_node->transformToOtherChildren(special_action, logprobs, parent_mst);

  return special_child_node;
}

template<typename T>
T *NodeAllocator<T>::split(InfoNode *parent, int special_child, float cost) {
  // Modifies `parent` in place. Assumes it's not needed any more.
  InfoNode *other_children_node = parent;

  InfoNode *special_child_node = getNew();
  special_child_node->CopyState(parent);

  special_child_node->transformToSpecialChild(special_child, cost);
  other_children_node->transformToOtherChildren(special_child);

  return special_child_node;
}
#endif  // NODE_ALLOCATOR_H__
