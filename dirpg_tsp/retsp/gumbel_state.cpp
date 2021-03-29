#include "gumbel_state.h"

#include <numeric>
#include <math.h>
//#include <iostream>

//namespace py = pybind11;
default_random_engine global_generator;

GumbelState::GumbelState(int graph_size):
expo_(1.0),generator_(global_generator)
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

float GumbelState::partialLogSumExp(vector<float> logprobs, vector<char> legal_next_actions){
    float logsumexp = 0;
    //clock_t before_data, before_for, after_for;

    //before_data = clock();
    //float *logprobs_arr = logprobs.data<float>();
    //before_for = clock();

    //double data_time = double (before_for - before_data);

    for (int i=0; i<logprobs.size(); i++){
        if (legal_next_actions[i])
            logsumexp += exp(logprobs[i]);

    }
    //after_for = clock();
    //double for_time = double (after_for - before_for);

    //py::print("data time: ", data_time/ (data_time+for_time));
    //py::print("for time: ", for_time/ (data_time+for_time));
    return log(logsumexp);
}


void GumbelState::transformToSpecialChild(int special_action, vector<float> logprobs) {
  /* Make this node into a special child of its current values. */
  //logprob_so_far_ += logprobs.index({special_action});
  logprob_so_far_ += logprobs[special_action];
}


void GumbelState::transformToOtherChildren(int special_action, vector<float> logprobs, MstNode *parent_mst) {
  /* Make this node into the other children of its current values.
  //torch::Tensor next_actions(parent_mst -> getLegalNextActionMask());
  max_gumbel_ = sample_truncated_gumbel(
                logprob_so_far_ + torch::logsumexp(logprobs.masked_select(parent_mst-> getLegalNextActionMask()),0).item<float>(),
                max_gumbel_);
  */
  //clock_t before_logsumexp, before_sample_truncated, after_sample_truncated;
  //before_logsumexp = clock();

  float logsumexp = partialLogSumExp(logprobs, parent_mst-> getLegalNextActionMask());

  //before_sample_truncated = clock();
  //double lgse_time = double (before_sample_truncated - before_logsumexp);
  //py::print("-------------------");
  //py::print("logprob_so_far_: ", logprob_so_far_);
  //py::print("logsumexp: ", logsumexp);
  //py::print("max gumbel: ", max_gumbel_);

  max_gumbel_ = sample_truncated_gumbel(
                logprob_so_far_ + logsumexp,
                max_gumbel_);
  //py::print("max gumbel after: ", max_gumbel_);
  //after_sample_truncated = clock();
  //double trunc_time = double (after_sample_truncated - before_sample_truncated);

  //py::print("logsumexp time: ", lgse_time/ (lgse_time+trunc_time));
  //py::print("sample_truncated_gumbel time: ", trunc_time/ (lgse_time+trunc_time));
  //py::print(max_gumbel_);
  //max_gumbel_ = gumbel_generator.sample_truncated_gumbel(logprob_so_far_ + logsumexp(logprobs.index({next_actions})), max_gumbel_);
}
