#include "ATen/Parallel.h"
#include <torch/extension.h>
#include <iostream>

#include <memory>


#include "retsp/a_star_sampling.h"
#include "retsp/mst_node.h"
#include "retsp/node_allocator.h"

//#include <pybind11/stl.h>
//#include <pybind11/complex.h>
//#include <pybind11/functional.h>
namespace py = pybind11;

using namespace std;

/*
int dannys_main() {

  UnionFind uf = UnionFind(10);
  //uf.dump();

  uf.merge(0, 1);
  //uf.dump();

  uf.merge(1, 2);
  //uf.dump();

  uf.merge(3, 4);
  //uf.dump();

  uf.merge(1, 3);
  //uf.dump();


  int batch_size = 3;
  int graph_size = 4;

  float weights0[] = { 0, .1, .8, .1,
                       0,  0, .8, .1,
                       0,  0,  0, .8,
                       0,  0,  0,  0 };

  float weights1[] = { 0, .8, .1, .1,
                       0,  0, .8, .8,
                       0,  0,  0, .1,
                       0,  0,  0,  0 };

  float weights2[] = { 0, .1, .1, .8,
                       0,  0, .1, .8,
                       0,  0,  0, .8,
                       0,  0,  0,  0 };

  BatchedGraphs graphs = BatchedGraphs(batch_size, graph_size);
  graphs.setWeights(0, weights0);
  graphs.setWeights(1, weights1);
  graphs.setWeights(2, weights2);
  //graphs.dump();

  py::print("MST costs:");
  for (int b = 0; b < batch_size; b++) {
    py::print(graphs.mstCost(b));
  }

  int search_budget = 10;  // Maximum number of nodes we need for search.
  NodeAllocator<MstNode> node_allocator = NodeAllocator<MstNode>(search_budget, graph_size); // should be called from dirpg.py
  NodeAllocator<GumbelState> gumbel_node_allocator = NodeAllocator<GumbelState>(search_budget, graph_size);
  // Suppose we've visited node 2 and can only go to 1 next. I.e., we're
  // an "other" child and the special children 0 and 3 were already chosen.
  bool visited_mask0_[] = { false, false, true, false };
  bool legal_next_action_mask0_[] = { false, true, false, false };
  MstNode *mst_node_ = node_allocator.getNew();

  mst_node_->setVisitedMask(visited_mask0_);
  mst_node_->setLegalNextActionMask(legal_next_action_mask0_);
  mst_node_->setFirstLast(2, 0);

  mst_node_->dump();

  py::print("MST costs after partial tour:");
  for (int b = 0; b < batch_size; b++) {
    py::print(graphs.mstCost(b, *mst_node_) );
  }
  // Try splitting.
  int special_child_ = 1;

  MstNode *special_child_node_ = node_allocator.split(mst_node_, special_child_);
  py::print("Special child:");
  special_child_node_->dump();

  py::print("Other children:");
  mst_node_->dump();





  py::print("******************************");
  bool visited_mask0[] = { false, false, false, false };
  bool legal_next_action_mask0[] = { true, true, true, false };

  MstNode *mst_node = node_allocator.getNew();
  mst_node->setVisitedMask(visited_mask0);
  mst_node->setLegalNextActionMask(legal_next_action_mask0);
  mst_node->setFirstLast(3, 3);

  //py::print(mst_node);

  py::print("MST costs after partial tour:");
  for (int b = 0; b < batch_size; b++) {
    py::print(graphs.mstCost(b, *mst_node));
  }


  // Try splitting.
   py::print("start splitting");

   int special_actions[] = {1,0,2};
   for (int i=0; i < 3; i++){
        py::print("---------------------------");
        py::print("MST costs after  ",special_actions[i],  "  action partial tour:");
        MstNode *special_child_node = node_allocator.split(mst_node, special_actions[i]);
        py::print("Special child:");
        special_child_node->dump();
        py::print("special_child_node mst");
        for (int b = 0; b < batch_size; b++) {
            py::print(graphs.mstCost(b, *special_child_node));
        }

        py::print("Other children:");
        mst_node->dump();
        py::print("Other children mst");
        for (int b = 0; b < batch_size; b++) {
            py::print(graphs.mstCost(b, *mst_node));
      }
  }
  //int special_child_new = 2;
  return 0;
}

void fill_graphs(BatchedGraphs& graphs, torch::Tensor weights){
    int batch_size = weights.size({0});
    int twice_graph_size = weights.size({1});

    for (int i=0; i < batch_size; i++){
        float dm[twice_graph_size];
        for(int idx=0; idx <twice_graph_size; idx++){
            dm[idx] = weights[i][idx].item<float>();
        }
        graphs.setWeights(i, dm);
    }
}

vector<float> compute_mst(BatchedGraphs& graphs, int batch_size){
    vector<float> mst_vals;
    for (int b = 0; b < batch_size; b++){
       mst_vals.push_back(graphs.mstCost(b));
    }
    return mst_vals;
}

*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  py::class_<AstarSampling>(m, "AstarSampling")
      .def(py::init<int, int, int, float, float,bool, int>())
      .def("initialize", &AstarSampling::initialize)
      .def("expand", &AstarSampling::expand)
      .def("popBatch", &AstarSampling::popBatch)
      .def("getTrajectories", &AstarSampling::getTrajectories)
      .def("getNonEmptyHeaps", &AstarSampling::getNonEmptyHeaps)
      .def("clear", &AstarSampling::clear);

  py::class_<EnvInfo>(m, "EnvInfo")
      //.def(py::init<HeapNode>())
      .def_readonly("batch_t", &EnvInfo::batch_t)
      .def_readonly("batch_prev_city", &EnvInfo::batch_prev_city)
      .def_readonly("batch_next_actions", &EnvInfo::batch_next_actions);

  py::class_<EmptyHeapsFilter>(m, "EmptyHeapsFilter")
      .def_readonly("non_empty_heaps", &EmptyHeapsFilter::non_empty_heaps);
      //.def_readonly("active_heaps", &EnvInfo::active_heaps);

  py::class_<BatchedTrajectories>(m, "BatchedTrajectories")
      //.def(py::init<HeapNode>())
      .def_readonly("costs", &BatchedTrajectories::costs)
      .def_readonly("objectives", &BatchedTrajectories::objectives)
      .def_readonly("actions", &BatchedTrajectories::actions);



  py::class_<ToptTdirect>(m, "ToptTdirect")
      //.def(py::init<HeapNode>())
      .def("get_t_opt_direct", &ToptTdirect::get_t_opt_direct)
      .def_readonly("t_opt", &ToptTdirect::t_opt)
      .def_readonly("t_direct", &ToptTdirect::t_direct)
      .def_readonly("prune_count", &ToptTdirect::prune_count)
      .def_readonly("num_candidates", &ToptTdirect::num_candidates);
}
