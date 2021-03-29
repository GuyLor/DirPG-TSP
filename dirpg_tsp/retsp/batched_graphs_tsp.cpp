#include "batched_graphs_tsp.h"

#include <algorithm>
#include "assert.h"

#include "mst_node.h"

#define V_MAX 9999.0
/*
ostream& operator<<(ostream& os, const WeightedEdge& edge) {
  return os << "(" << edge.src << ", " << edge.dest << "):" << edge.weight;
}
*/


BatchedGraphsOptimal::BatchedGraphsOptimal(int batch_size, int graph_size) :
  batch_size_(batch_size),
  graph_size_(graph_size)
  {
  // Initialize storage for graph edge weights.
  _graphs.resize(batch_size);
  for (int i = 0; i < batch_size_; i++) {
    _graphs[i].resize(graph_size_);
    for(int j = 0; j < graph_size_; j++) {
        _graphs[i][j].resize(graph_size_);
        }
    }
  }

//BatchedGraphsOptimal::~BatchedGraphsOptimal() {}

void BatchedGraphsOptimal::setWeights(int batch_index, float *weights) {
  /* Stores weights as WeightedEdges and sorts by edge weight.
   *
   * Assumes `weights` is a `graph_size_ * graph_size_` array of a dense
   * adjacency matrix, and the cost from `i` to `j` is at position
   * weights[i * graph_size_ + j].
   *
   * Copies values out of `weights` into pre-allocated memory being held
   * in `_graphs`.
   */
  vector<vector<float>> &graph = _graphs[batch_index];
  for (int i = 0; i < graph_size_; i++) {
    for (int j = 0; j < graph_size_; j++) {
        graph[i][j] = weights[i * graph_size_ + j];
    }
  }
}


void BatchedGraphsOptimal::dump(int batch_index) {
    for (const auto cities : _graphs[batch_index]) {
      for (int i = 0; i < graph_size_; i++) {
        py::print(cities[i]);
        }
      }
}

float BatchedGraphsOptimal::tspCost(int batch_index, MstNode &mst_node){
    const vector<vector<float>>& graph = _graphs[batch_index];

    if (mst_node.getNumVisited() < graph_size_){
        //py::print("solver!!");
        solver.initialize(mst_node.prepareTSPdistanceMatrix(graph));
        return solver.tsp(1,0);
    }
    else{
        //py::print("last step!!");
        return mst_node.computeLastStep(graph);
    }



}

TSP_Solver::TSP_Solver(){

}

void TSP_Solver::initialize(vector<vector<float>> dist_){
    dist = dist_;
    ///py::print("------ dist --------");

    ///for (auto i : dist){
    ///    py::print(i);
    ///}
    n = dist_.size();
    VISITED_ALL = (1<<n) -1;

    dp.resize(1<<n);
    for (int i=0; i< (1<<n); i++){
        dp[i].clear();
        for (int j=0; j<n; j++)
            dp[i].push_back(-1);
        }
}

float  TSP_Solver::tsp(int mask,int pos){
    if(mask==VISITED_ALL){
        return dist[pos][0];
    }
    if(dp[mask][pos]!=-1){
    return dp[mask][pos];
    }

    //Now from current node, we will try to go to every other node and take the min ans
    float ans = V_MAX;

    //Visit all the unvisited cities and take the best route
    for(int city=0;city<n;city++){

        if((mask&(1<<city))==0){

            float newAns = dist[pos][city] + tsp( mask|(1<<city), city);
            ans = min(ans, newAns);
        }

    }
    dp[mask][pos] = ans;
    return ans;
}
