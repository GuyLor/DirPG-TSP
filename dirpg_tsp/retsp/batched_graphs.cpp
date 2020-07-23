#include "batched_graphs.h"

#include <algorithm>
#include "assert.h"

#include "mst_node.h"

/*
ostream& operator<<(ostream& os, const WeightedEdge& edge) {
  return os << "(" << edge.src << ", " << edge.dest << "):" << edge.weight;
}
*/

bool operator<(const WeightedEdge& a, const WeightedEdge& b) {
    return a.weight < b.weight;
}

BatchedGraphs::BatchedGraphs(int batch_size, int graph_size) :
  batch_size_(batch_size),
  graph_size_(graph_size),
  _uf(graph_size)
{
  // Initialize storage for graph edge weights.
  _graphs.resize(batch_size);
  int num_edges = (graph_size * (graph_size - 1)) / 2;
  for (int i = 0; i < batch_size_; i++) {
    _graphs[i].resize(num_edges);
  }
}


BatchedGraphs::~BatchedGraphs() {}



void BatchedGraphs::setWeights(int batch_index, float *weights) {
  /* Stores weights as WeightedEdges and sorts by edge weight.
   *
   * Assumes `weights` is a `graph_size_ * graph_size_` array of a dense
   * adjacency matrix, and the cost from `i` to `j` is at position
   * weights[i * graph_size_ + j].
   *
   * Copies values out of `weights` into pre-allocated memory being held
   * in `_graphs`.
   */
  vector<WeightedEdge> &graph = _graphs[batch_index];
  int edge_position = 0;
  for (int i = 0; i < graph_size_; i++) {
    for (int j = i + 1; j < graph_size_; j++) {
      WeightedEdge &edge = graph[edge_position++];
      edge.src = i;
      edge.dest = j;
      edge.weight = weights[i * graph_size_ + j];
    }
  }
  sort(graph.begin(), graph.end());
}


void BatchedGraphs::dump(int batch_index) {

    for (const auto &edge : _graphs[batch_index]) {
      py::print("(", edge.src,", ", edge.dest,"): ", edge.weight);
    }
}



float BatchedGraphs::mstCost(int batch_index) {
  /* Runs Kruskal's algorithm to compute MST cost for one graph.
   */
  _uf.reset();

  float total_cost = 0;
  int num_edges_used = 0;
  for (const auto& edge : _graphs[batch_index]) {
    int src_group = _uf.findRoot(edge.src);
    int dest_group = _uf.findRoot(edge.dest);

    if (src_group != dest_group) {
      _uf.merge(edge.src, edge.dest);
      total_cost += edge.weight;
      num_edges_used++;

      if (num_edges_used == graph_size_ - 1)  break;
    }
  }

  assert(num_edges_used == graph_size_ - 1);
  return total_cost;
}



float BatchedGraphs::mstCost(int batch_index, const MstNode &cpp_node) {
  /* Runs Kruskal's algorithm to compute MST cost for one graph.
   */
  _uf.reset();

  float total_cost = 0;
  int num_edges_used = 0;
  //py::print("gfgfgf");

  for (const auto& edge : _graphs[batch_index]) {
    // Information in `cpp_node` determines whether we can use each edge.
    if (!cpp_node.canUseEdge(edge.src, edge.dest)) continue;

    int src_group = _uf.findRoot(edge.src);
    int dest_group = _uf.findRoot(edge.dest);

    if (src_group != dest_group) {
      _uf.merge(edge.src, edge.dest);
      //py::print(edge.weight);
      total_cost += edge.weight;
      num_edges_used++;

      if (num_edges_used == graph_size_ - 1)  break;
    }
  }
  //py::print("fkfkfkf");
  assert(num_edges_used == graph_size_ - 1);
  return total_cost;
}
