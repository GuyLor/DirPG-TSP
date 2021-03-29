/* A search node that only stores information needed by the c++ side.
 *
 * We need information to compute spanning trees and to split CppNodes.
 * Otherwise, don't put the data into here.
 */

#ifndef GUMBEL_STATE_H__
#define GUMBEL_STATE_H__

#include <vector>
#include <torch/extension.h>
#include <random>
#include "mst_node.h"

using namespace std;

//using namespace torch::indexing;
/*
float logsumexp(double arr[], int count){
   if(count > 0 ){
      float maxVal = arr[0];
      float sum = 0;

      for (int i = 1 ; i < count ; i++){
         if (arr[i] > maxVal){
            maxVal = arr[i];
         }
      }

      for (int i = 0; i < count ; i++){
         sum += exp(arr[i] - maxVal);
      }
      return log(sum) + maxVal;
   }
   else
   {
      return 0.0;
   }
}
*/

class GumbelState {

 public:
  GumbelState(int graph_size);

  void CopyState(const GumbelState *other_node);
  void setRoot(default_random_engine &generator);

  // Print out state, just for debugging.
  void dump() const;

  // Expansions. We'll re-use memory when possible, so define expansion in terms
  // of converting existing nodes into the new ones we need.

  void transformToSpecialChild(int special_action, vector<float> logprobs);
  void transformToOtherChildren(int special_action, vector<float> logprobs, MstNode *parent_mst);
  void setMaxGumbelLogprobs(float max_gumbel, float logprob_so_far);
  float getMaxGumbel();
  float sample_gumbel(float location=0.0, float scale=1.0);
  float sample_truncated_gumbel(float location, float b);


 protected:

  float max_gumbel_;
  float logprob_so_far_;
  float partialLogSumExp(vector<float> logprobs, vector<char> legal_next_actions);
  default_random_engine &generator_;
  exponential_distribution<float> expo_;

};


#endif  // CPP_NODE_H__
