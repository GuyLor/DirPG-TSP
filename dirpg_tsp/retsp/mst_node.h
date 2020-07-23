/* A search node that only stores information needed by the c++ side.
 *
 * We need information to compute spanning trees and to split CppNodes.
 * Otherwise, don't put the data into here.
 */

#ifndef CPP_NODE_H__
#define CPP_NODE_H__

#include <algorithm>
#include <vector>
#include <torch/extension.h>
using namespace std;


class MstNode {

 public:
  MstNode(int graph_size);

  void setRoot(int start_city);
  void CopyState(const MstNode *other_node);

  // Copies the input array by value.
  void setVisitedMask(char *visited_mask);

  void setVisitedMask(const vector<char> &visited_mask_begin);

  // Copies the input array by value.
  void setLegalNextActionMask(char *legal_next_action_mask);

  void setLegalNextActionMask(const vector<char> &legal_next_action_mask);

  void setFirstLast(int first, int last);

  int getPrevCity();

  bool isAnyLegalAction() const;

  torch::Tensor getLegalNextActionMask();

  bool canUseEdge(int i, int j) const;

  // Print out state, just for debugging.
  void dump() const;

  // Expansions. We'll re-use memory when possible, so define expansion in terms
  // of converting existing nodes into the new ones we need.
  void transformToSpecialChild(int special_child);
  void transformToOtherChildren(int special_child);

 protected:
  void setAllAreLegalNextActions();


  int graph_size_;

  // `graph_size_` vector of indicators specifying if a node has been visited.
  vector<char> visited_mask_;

  // `graph_size_` vector of indicators specifying whether its legal for the
  // next edge to connect `first_node_` to each other node.
  vector<char> legal_next_action_mask_;

  // Integer indices of the first node in the remainder of the tour (most
  // recently visited) and the last node (where the tour should end).
  int first_node_;
  int last_node_;

};


#endif  // CPP_NODE_H__
