/* A search node that only stores information needed by the c++ side.
 *
 * We need information to compute spanning trees and to split CppNodes.
 * Otherwise, don't put the data into here.
 */

#ifndef INFO_NODE_H__
#define INFO_NODE_H__

#include <vector>

using namespace std;


class InfoNode {

 public:
  InfoNode(int graph_size);

  void setRoot(int start_city);
  void CopyState(const InfoNode *other_node);
  void setPrefix(int *prefix);
  void setPrefix(const vector<int> &prefix);
  void setLastStepCost(int *last_step_cost);
  void setLastStepCost(const vector<float> &last_step_cost);
  void setCost(float cost);
  vector<int> getActions();
  float getCost();
  bool getIsTopt();
  bool getIsDone();
  int getT();
  int getGraphSize();
  // Print out state, just for debugging.
  void dump() const;

  // Expansions. We'll re-use memory when possible, so define expansion in terms
  // of converting existing nodes into the new ones we need.
  void transformToSpecialChild(int special_child, float cost);
  void transformToOtherChildren(int special_child);

 protected:


  int graph_size_;
  vector<int> prefix_;
  vector<float> last_step_cost_;
  float cost_so_far_;
  int t_;
  bool is_t_opt_;
  bool done_;


};


#endif  // CPP_NODE_H__
