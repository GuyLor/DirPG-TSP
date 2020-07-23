#ifndef BATCHED_GRAPHS_H__
#define BATCHED_GRAPHS_H__

#include <iostream>
#include <vector>

#include "union_find.h"


using namespace std;

class MstNode;


struct WeightedEdge {
  float weight;
  int src;
  int dest;

  WeightedEdge() :
    weight(0),
    src(-1),
    dest(-1) {};

  WeightedEdge(float other_weight, int other_src, int other_dest) :
    weight(other_weight),
    src(other_src),
    dest(other_dest) {}
};

//ostream& operator<<(ostream& os, const WeightedEdge& edge);

bool operator<(const WeightedEdge& a, const WeightedEdge& b);


class BatchedGraphs {

public:
  BatchedGraphs(int batch_size, int graph_size);
  ~BatchedGraphs();

  void setWeights(int batch_index, float *weights);

  void dump(int batch_index);

  float mstCost(int batch_index);

  float mstCost(int batch_index, const MstNode &cpp_node);


protected:
  int batch_size_;
  int graph_size_;

  vector<vector<WeightedEdge> > _graphs;
  UnionFind _uf;
};


#endif  // BATCHED_GRAPHS__
