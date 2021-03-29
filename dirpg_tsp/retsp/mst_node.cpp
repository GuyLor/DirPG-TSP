#include "mst_node.h"

//#include <iostream>

//namespace py = pybind11;
#include <torch/extension.h>
#include <algorithm>
#include <functional>
#define V_MAX 9999.0

MstNode::MstNode(int graph_size) :
  graph_size_(graph_size)
{
  visited_mask_.resize(graph_size);
  legal_next_action_mask_.resize(graph_size);
}

void MstNode::setRoot(int start_city){

    vector<char> visited_mask0(graph_size_, false);
    visited_mask0[start_city] = true;
    num_visited_ = 1;
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
  num_visited_ = other_node->num_visited_;
}

int MstNode::getPrevCity(){
    return first_node_;
}

void MstNode::setVisitedMask(vector<bool>  visited_mask) {
  // `visited_mask` must be an array of length `graph_size_`.
  for (int i=0; i<visited_mask.size(); i++){
    visited_mask_[i] = (char)visited_mask[i];
  }
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

void MstNode::setLegalNextActionMask(vector<bool> legal_next_action_mask) {
  // `legal_next_action_mask` must be an array of length `graph_size_`.
  for (int i=0; i<legal_next_action_mask.size(); i++){
      legal_next_action_mask_[i] = (char)legal_next_action_mask[i];
  }
}


bool MstNode::isAnyLegalAction() const{
    return all_of(legal_next_action_mask_.begin(),
                  legal_next_action_mask_.end(),
                  [](char i) { return i==0; });
}

torch::Tensor MstNode::getLegalNextActionMaskTorch(){
    auto opts = torch::TensorOptions().dtype(torch::kUInt8); //kBool
    return torch::from_blob(legal_next_action_mask_.data(), {graph_size_}, opts).to(torch::kBool);
}

vector<char> MstNode::getLegalNextActionMask(){
    return legal_next_action_mask_;
}

vector<char> MstNode::getVisitedMask(){
    return visited_mask_;
}

void MstNode::setAllAreLegalNextActions() {
  fill(legal_next_action_mask_.begin(),
       legal_next_action_mask_.end(),
       true);
}

int MstNode::getNumVisited(){
    return num_visited_;
}

void MstNode::setNumVisited(int num) {
  num_visited_ = num;
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

  //if (visited_mask_[i] && visited_mask_[j] && ((i==first_node_ && j==last_node_)||(j==first_node_ && i==last_node_))) {
  //  return true;
  //}

  // If both have been visited, then we definitely don't want to use.
  if (visited_mask_[i] && visited_mask_[j]) {
    return false;
  }
  // Otherwise, exactly one has been visited.
  // 1. the visited node is the root node (last node)
  // 2. the visited node is the current node (first node)
  // 3. the visited node is in the prefix (w/o root,current)

  // 1
  if (visited_mask_[i] && i == last_node_){
    return true;
  }

  if (visited_mask_[j] && j == last_node_){
    return true;
  }

  // 3
  if (visited_mask_[i] && i != first_node_ &&  i != last_node_){
    return false;
  }

   if (visited_mask_[j] && j != first_node_ && j != last_node_){
    return false;
  }

  if (visited_mask_[i]) {
    return legal_next_action_mask_[j];
  }

  if (visited_mask_[j]) {
    return legal_next_action_mask_[i];

  }

  py::print("BUG canUseEdge");
  return false;

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

vector<vector<float>> MstNode::prepareTSPdistanceMatrix(const vector<vector<float>>& full_dm){

vector<vector<float>> dm;
for (int i=0; i<graph_size_+1; i++){

    if (i != graph_size_)
        if (visited_mask_[i] && i != last_node_ && i != first_node_)
            continue;

    if((first_node_ == last_node_) && (i==graph_size_ ))
        continue;

    vector<float> row;
    for(int j=0; j<graph_size_+1; j++){
        if (i==j && (j!=graph_size_ ||  i!=graph_size_)){
            row.push_back(0);
            continue;
        }
        // removing the prefix (non-first-last-visited)
        if (j < visited_mask_.size() && visited_mask_[j] && j != last_node_ && j != first_node_){
            continue;
        }
        else if (i < visited_mask_.size() && visited_mask_[i] && i != last_node_ && i != first_node_){
            continue;
        }

        else if((first_node_ != last_node_) && (j==graph_size_ || i==graph_size_ )){
            // connect the dummy node with weight 0 to first and last
            if ( i == first_node_ || i == last_node_){
                 row.push_back(0);
            }

            else if (j == first_node_ || j == last_node_){
                 row.push_back(0);
            }
            // connect the dummy node to the rest of the nodes with Inf
            else if (i==graph_size_ || j==graph_size_){
                row.push_back(V_MAX);
            }
        }
        else if ((first_node_ == last_node_) && (j==graph_size_ || i==graph_size_ )){
            // connect the dummy node with weight 0 to first and last
            continue;
            if (j==graph_size_ && (i == first_node_ || i == last_node_)){
                 row.push_back(V_MAX);
            }

            else if (i==graph_size_ && (j == first_node_ || j == last_node_)){
                 row.push_back(V_MAX);
            }
            // connect the dummy node to the rest of the nodes with Inf
            else if (i==graph_size_ || j==graph_size_){
                row.push_back(0);
            }
        }

        else if (visited_mask_[j] && j == first_node_ && !legal_next_action_mask_[i]){
            row.push_back(V_MAX);
            }
        else if (visited_mask_[i] && i == first_node_ && !legal_next_action_mask_[j]){
            row.push_back(V_MAX);
        }
        else{
            row.push_back(full_dm[i][j]);
        }
    }
    dm.push_back(row);

}
return dm;
}

float MstNode::computeLastStep(const vector<vector<float>>& full_dm){
    return full_dm[first_node_][last_node_];
}
void MstNode::transformToSpecialChild(int special_child) {
  /* Make this node into a special child of its current values. */
  visited_mask_[special_child] = true;


  vector<char> legal_next_actions(graph_size_, true);
  for (int i=0; i<graph_size_; i++){
    //if (i==last_node_ && num_visited_>1)
    //   legal_next_actions[i] = true;
    //else
    legal_next_actions[i] = !visited_mask_[i];
  }
  setLegalNextActionMask(legal_next_actions);
  //legal_next_action_mask_[special_child] = false;

  first_node_ = special_child;
  num_visited_ += 1;
  //setAllAreLegalNextActions();
}


void MstNode::transformToOtherChildren(int special_child) {
  /* Make this node into the other children of its current values. */
  legal_next_action_mask_[special_child] = false;
}
