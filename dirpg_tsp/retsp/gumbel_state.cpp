#include "gumbel_state.h"

#include <numeric>
#include <math.h>
//#include <iostream>

//namespace py = pybind11;


GumbelState::GumbelState(int graph_size):
expo_(1.0)
{
}

void GumbelState::setRoot(default_random_engine &generator){
  generator_ = generator;
  max_gumbel_ = sample_gumbel();
  logprob_so_far_ = 0.0;
}

void GumbelState::CopyState(const GumbelState *other_node) {
  max_gumbel_ = other_node->max_gumbel_;
  logprob_so_far_ = other_node->logprob_so_far_;
}

void GumbelState::setMaxGumbelLogprobs(float max_gumbel, float logprob_so_far) {
  max_gumbel_ = max_gumbel;
  logprob_so_far_ = logprob_so_far;
}
float GumbelState::getMaxGumbel(){
    return max_gumbel_;
  }
void GumbelState::dump() const {
  py::print("max gumbel: ", max_gumbel_);
  py::print("logprob so far: ", logprob_so_far_);
}

float GumbelState::sample_gumbel(float location, float scale){
    return -log(expo_(generator_)) + location;
}
float GumbelState::sample_truncated_gumbel(float location, float b){
    //Sample a Gumbel(location) truncated to be less than b
    return -log(expo_(generator_) + exp(-b + location)) + location;
}
void GumbelState::transformToSpecialChild(int special_action, torch::Tensor logprobs) {
  /* Make this node into a special child of its current values. */
  //logprob_so_far_ += logprobs.index({special_action});
  logprob_so_far_ += logprobs[special_action].item<float>();
}


void GumbelState::transformToOtherChildren(int special_action, torch::Tensor logprobs, MstNode *parent_mst) {
  /* Make this node into the other children of its current values. */
  //torch::Tensor next_actions(parent_mst -> getLegalNextActionMask());
  max_gumbel_ = sample_truncated_gumbel(
                logprob_so_far_ + torch::logsumexp(logprobs.masked_select(parent_mst -> getLegalNextActionMask()),0).item<float>(),
                max_gumbel_);
  //py::print(max_gumbel_);
  //max_gumbel_ = gumbel_generator.sample_truncated_gumbel(logprob_so_far_ + logsumexp(logprobs.index({next_actions})), max_gumbel_);
}
