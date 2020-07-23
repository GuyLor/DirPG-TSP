#include "info_node.h"

#include <random>
#include <numeric>
//#include <iostream>

//namespace py = pybind11;
#include <torch/extension.h>

InfoNode::InfoNode(int graph_size) :
  graph_size_(graph_size)
{
  prefix_.resize(graph_size);
  //last_step_cost_.resize(graph_size);
}

void InfoNode::setRoot(int start_city){
  vector<int> root_prefix(graph_size_, -1);
  root_prefix[0] = start_city;
  setPrefix(root_prefix);
  cost_so_far_ = 0.0;
  t_ = 0;
  is_t_opt_ = true;
  done_ = false;
}

void InfoNode::CopyState(const InfoNode *other_node) {
  setPrefix(other_node->prefix_);
  //setLastStepCost(other_node->last_step_cost_);
  cost_so_far_ = other_node->cost_so_far_;
  t_ = other_node->t_;
  is_t_opt_ = other_node->is_t_opt_;
  done_ = other_node->done_;
}


void InfoNode::setPrefix(int *prefix) {
  // `visited_mask` must be an array of length `graph_size_`.
  copy(prefix, prefix + graph_size_, prefix_.begin());
}

void InfoNode::setPrefix(const vector<int> &prefix) {
  // `visited_mask` must be an array of length `graph_size_`.
  copy(prefix.begin(), prefix.end(), prefix_.begin());
}
/*
void InfoNode::setLastStepCost(int *last_step_cost) {
  // `visited_mask` must be an array of length `graph_size_`.
  copy(last_step_cost, last_step_cost + graph_size_, last_step_cost_.begin());
}

void InfoNode::setLastStepCost(const vector<float> &last_step_cost) {
  // `visited_mask` must be an array of length `graph_size_`.
  copy(last_step_cost.begin(), last_step_cost.end(), last_step_cost_.begin());
}
*/
void InfoNode::setCost(float cost) {
  cost_so_far_ = cost;
}

vector<int> InfoNode::getActions(){
    return prefix_;
}
float InfoNode::getCost(){
    return cost_so_far_;
}

bool InfoNode::getIsTopt(){
    return is_t_opt_;
}

bool InfoNode::getIsDone(){
    return done_;
}

int InfoNode::getT(){
    return t_;
}

int InfoNode::getGraphSize(){
    return graph_size_;
}

void InfoNode::dump() const {
  py::print("prefix: ");
  py::print(prefix_);
  //for (const auto &val : prefix_) {
  //  py::print(val) ;
  //}

  py::print("cost_so_far: " ,cost_so_far_);
  py::print("t: " ,t_);
  }

void InfoNode::transformToSpecialChild(int special_child, float cost) {
  /* Make this node into a special child of its current values. */
  t_ += 1;
  prefix_[t_] = special_child;
  cost_so_far_ += cost;
  if (prefix_.back() != -1){
    done_ = true;
  }
}


void InfoNode::transformToOtherChildren(int special_child) {
    is_t_opt_ = false;
  /* Make this node into the other children of its current values. */
}