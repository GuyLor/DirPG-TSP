#include "mst_node.h"

//#include <iostream>

//namespace py = pybind11;
#include <torch/extension.h>
#include <algorithm>
#include <functional>

MstNode::MstNode(int graph_size) :
  graph_size_(graph_size)
{
  visited_mask_.resize(graph_size);
  legal_next_action_mask_.resize(graph_size);
}

void MstNode::setRoot(int start_city){

    vector<char> visited_mask0(graph_size_, false);
    visited_mask0[start_city] = true;

    vector<char> legal_next_action_mask0(graph_size_, true);
    legal_next_action_mask0[start_city] = false;
    setVisitedMask(visited_mask0);
    setLegalNextActionMask(legal_next_action_mask0);
    setFirstLast(start_city, start_city);
}


void MstNode::CopyState(const MstNode *other_node) {
  setVisitedMask(other_node->visited_mask_);
  setLegalNextActionMask(other_node->legal_next_action_mask_);
  first_node_ = other_node->first_node_;
  last_node_ = other_node->last_node_;
}

int MstNode::getPrevCity(){
    return first_node_;
}

void MstNode::setVisitedMask(char *visited_mask) {
  // `visited_mask` must be an array of length `graph_size_`.
  copy(visited_mask, visited_mask + graph_size_, visited_mask_.begin());
}

void MstNode::setVisitedMask(const vector<char> &visited_mask) {
  // `visited_mask` must be an array of length `graph_size_`.
  copy(visited_mask.begin(), visited_mask.end(), visited_mask_.begin());
}

void MstNode::setLegalNextActionMask(char *legal_next_action_mask) {
  // `legal_next_action_mask` must be an array of length `graph_size_`.
  copy(legal_next_action_mask,
       legal_next_action_mask + graph_size_,
       legal_next_action_mask_.begin());
}

void MstNode::setLegalNextActionMask(const vector<char> &legal_next_action_mask) {
  // `legal_next_action_mask` must be an array of length `graph_size_`.

  copy(legal_next_action_mask.begin(),
       legal_next_action_mask.end(),
       legal_next_action_mask_.begin());
}


bool MstNode::isAnyLegalAction() const{
    return all_of(legal_next_action_mask_.begin(),
                  legal_next_action_mask_.end(),
                  [](char i) { return i==0; });
}

torch::Tensor MstNode::getLegalNextActionMask(){
    auto opts = torch::TensorOptions().dtype(torch::kUInt8); //kBool
    return torch::from_blob(legal_next_action_mask_.data(), {graph_size_}, opts).to(torch::kBool);
}
void MstNode::setAllAreLegalNextActions() {
  fill(legal_next_action_mask_.begin(),
       legal_next_action_mask_.end(),
       true);
}

void MstNode::setFirstLast(int first, int last) {
  first_node_ = first;
  last_node_ = last;
}


bool MstNode::canUseEdge(int i, int j) const {
  /* Determines if it's legal for a spanning tree to use an edge i -- j.
   *
   * TODO: the logic here needs to be checked.
   */

  // If neither has been visited, then we definitely want to use.
  if (!visited_mask_[i] && !visited_mask_[j]) {
    return true;
  }

  // If both have been visited, then we definitely don't want to use.
  if (visited_mask_[i] && visited_mask_[j]) {
    return false;
  }

  // Otherwise, exactly one has been visited.
  if (visited_mask_[i] && i == first_node_) {
    return legal_next_action_mask_[j];

  } else if (visited_mask_[j] && j == first_node_) {
    return legal_next_action_mask_[i];

  } else {
    return false;
  }
}

void MstNode::dump() const {
  //py::print("isAnyLegalAction: ");
  //py::print(isAnyLegalAction());

  py::print("Visited mask: ");
  py::print(visited_mask_) ;
  //for (const auto &val : visited_mask_) {
  //  py::print((bool) val) ;
  //}


  py::print("Legal next action mask: ");
  py::print(legal_next_action_mask_) ;
  //for (const auto &val : legal_next_action_mask_) {
  // py::print((bool)val) ;
  //}

  py::print("First, last: " ,first_node_, ", " , last_node_) ;

  }

void MstNode::transformToSpecialChild(int special_child) {
  /* Make this node into a special child of its current values. */
  visited_mask_[special_child] = true;


  vector<char> legal_next_actions(graph_size_, true);
  for (int i=0; i<graph_size_; i++){
   legal_next_actions[i] = !visited_mask_[i];
  }
  setLegalNextActionMask(legal_next_actions);
  //legal_next_action_mask_[special_child] = false;

  first_node_ = special_child;
  //setAllAreLegalNextActions();
}


void MstNode::transformToOtherChildren(int special_child) {
  /* Make this node into the other children of its current values. */
  legal_next_action_mask_[special_child] = false;
}
