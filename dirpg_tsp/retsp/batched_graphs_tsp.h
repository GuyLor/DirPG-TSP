#ifndef BATCHED_GRAPHS_TSP_H__
#define BATCHED_GRAPHS_TSP_H__

#include <iostream>
#include <vector>


using namespace std;

class MstNode;

//ostream& operator<<(ostream& os, const WeightedEdge& edge);

struct TSP_Solver{

    vector<vector<float>> dist;
    vector<vector<float>> dp;
    int n;
    int VISITED_ALL;

    TSP_Solver();
    void initialize(vector<vector<float>> dist_);
    float  tsp(int mask, int pos);

};

class BatchedGraphsOptimal {

public:
  BatchedGraphsOptimal(int batch_size, int graph_size);
  //~BatchedGraphsOptimal();

  void setWeights(int batch_index, float *weights);

  void dump(int batch_index);

  float tspCost(int batch_index, MstNode &mst_node);


protected:
  int batch_size_;
  int graph_size_;
  TSP_Solver solver;

  vector<vector<vector<float>>> _graphs; // batch_size, graph_size, graph_size
};


#endif  // BATCHED_GRAPHS__
