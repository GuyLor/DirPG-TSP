#include "ATen/Parallel.h"
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <memory>

//#include <pybind11/stl.h>
//#include <pybind11/complex.h>
//#include <pybind11/functional.h>

using namespace std;

class BatchedGraphs{
    BatchedGraphs(torch::Tensor distance_matrices){

    };

};

struct NodeConverter;

struct HeapNode {
    float priority;
    bool t_opt;
    bool done;
    NodeConverter *converter;
    py::object python_node_object;
    int idx_original;

    ~HeapNode(){};
    HeapNode(pair<py::list, NodeConverter*> const node_args_pair){
        py::list node_args = node_args_pair.first;
        converter = node_args_pair.second;
        priority = node_args[0].cast<float>();
        t_opt = node_args[1].cast<bool>();
        done = node_args[2].cast<bool>();
        python_node_object = node_args[3];
        idx_original = node_args[4].cast<int>();
        //py::list batched_prefix(python_node_object.attr("prefix"));


    }
    HeapNode(const HeapNode &other){
        priority = other.priority;
        t_opt = other.t_opt;
        python_node_object = other.python_node_object;
        idx_original = other.idx_original;
        converter = other.converter;
        done = other.done;
    }
    /*
    HeapNode(HeapNode &&other){
        priority = other.priority;
        t_opt = other.t_opt;
        python_node_object = other.python_node_object;
        idx_original = other.idx_original;
        converter = other.converter;
        done = other.done;
    }
    */
    bool operator<(const HeapNode &other) const {
         if (t_opt == other.t_opt){
            return priority < other.priority;
         }
         if (t_opt && !other.t_opt){
            return false;
         }
         else{
            return true;
         }
    }
};

struct NodeConverter{

    torch::Tensor priority;
    torch::Tensor t_opt;
    torch::Tensor done;
    py::object python_object;
    torch::Tensor ids;

    torch::Tensor t;
    torch::Tensor next_actions;
    torch::Tensor not_visited;
    torch::Tensor prefix;
    torch::Tensor lengths;
    torch::Tensor cur_coord;

    torch::Tensor logprob_so_far;
    torch::Tensor bound_togo;
    torch::Tensor max_gumbel;

    NodeConverter(torch::Tensor _priority, torch::Tensor _t_opt, torch::Tensor _done, torch::Tensor _ids,
                  torch::Tensor _t, torch::Tensor _next_actions, torch::Tensor _not_visited, torch::Tensor _prefix,
                  torch::Tensor _lengths, torch::Tensor _cur_coord, torch::Tensor _logprob_so_far, torch::Tensor _bound_togo,
                  torch::Tensor _max_gumbel, py::object _python_object)
    {
        priority = _priority;
        t_opt = _t_opt;
        done = _done;
        ids = _ids;
        t = _t;
        next_actions = _next_actions;
        not_visited = _not_visited;
        prefix = _prefix;
        lengths = _lengths;
        cur_coord = _cur_coord;
        logprob_so_far = _logprob_so_far;
        bound_togo = _bound_togo;
        max_gumbel = _max_gumbel;
        python_object = _python_object;

    };

    pair<py::list, NodeConverter*> operator[] (int i) {
        // const NodeConverter *converter_object = this;
        py::list all_attrs;
        all_attrs.append(priority[i].item<float>());
        all_attrs.append(t_opt[i].item<bool>());
        all_attrs.append(done[i].item<bool>());
        all_attrs.append(python_object);
        all_attrs.append(ids[i].item<int>());

        pair<py::list,  NodeConverter*> ret(all_attrs, this);
        //ret = make_pair(all_attrs, this);
        return ret;
    }
};


struct Trajectory{
    int idx;
    py::object actions;  //this is actually a tensor
    float cost;
    float objective;

    Trajectory(HeapNode leaf_node){
        py::list batch_objective(leaf_node.python_node_object.attr("objective"));
        py::list batch_lengths(leaf_node.python_node_object.attr("lengths"));
        py::list batch_actions(leaf_node.python_node_object.attr("prefix"));

        idx = leaf_node.idx_original;
        objective = batch_objective[idx].cast<float>();
        cost = batch_lengths[idx].cast<float>();
        actions = batch_actions[idx];
    }
};

class Heap {
 private:
    std::vector<HeapNode> elements_;
 public:
    Heap(HeapNode new_node){

        elements_.push_back(new_node);
        push_heap(elements_.begin(), elements_.end());
    };
    void push(HeapNode new_node) {

        elements_.push_back(new_node);
        push_heap(elements_.begin(), elements_.end());
    }

    HeapNode pop() {
        HeapNode result = elements_.front();
        pop_heap(elements_.begin(), elements_.end());

        elements_.pop_back();
        return result;
    }

    bool is_empty() {
        return elements_.empty();
    };
};

void clearVectorContents( std::vector <HeapNode*> & a )
{
    for ( int i = 0; i < a.size(); i++ )
    {
        delete a[i];
    }
    a.clear();
}


pair<vector<torch::Tensor>, set<NodeConverter*>> stack_heap_nodes_vec(vector<HeapNode> heap_nodes,
                                                                      int batch_size,
                                                                      int graph_size){
    set<NodeConverter*> to_remove;
    torch::Tensor ids = torch::zeros({batch_size,1}, torch::dtype(torch::kInt64).requires_grad(false));
    torch::Tensor t = torch::zeros({batch_size,1}, torch::dtype(torch::kInt64).requires_grad(false));
    torch::Tensor next_actions = torch::zeros({batch_size,1,graph_size}, torch::dtype(torch::kUInt8).requires_grad(false));
    torch::Tensor not_visited = torch::zeros({batch_size,1,graph_size}, torch::dtype(torch::kUInt8).requires_grad(false));
    torch::Tensor prefix = torch::zeros({batch_size,graph_size}, torch::dtype(torch::kInt64).requires_grad(false));
    torch::Tensor lengths = torch::zeros({batch_size,1});
    torch::Tensor cur_coord = torch::zeros({batch_size,1,2});
    torch::Tensor done = torch::zeros({batch_size,1}, torch::dtype(torch::kUInt8).requires_grad(false));
    torch::Tensor logprob_so_far = torch::zeros({batch_size,1});
    torch::Tensor bound_togo = torch::zeros({batch_size,1});
    torch::Tensor max_gumbel = torch::zeros({batch_size,1});
    torch::Tensor is_t_opt = torch::zeros({batch_size}, torch::dtype(torch::kUInt8).requires_grad(false));

    for (int i=0; i < batch_size; i++){
        //NodeConverter conv = heap_nodes[i].converter;
        ids[i] = heap_nodes[i].converter -> ids[i];
        t[i] = heap_nodes[i].converter -> t[i];
        next_actions[i] = heap_nodes[i].converter -> next_actions[i];
        not_visited[i] = heap_nodes[i].converter -> not_visited[i];
        prefix[i] = heap_nodes[i].converter -> prefix[i];
        lengths[i] = heap_nodes[i].converter -> lengths[i];
        cur_coord[i] = heap_nodes[i].converter -> cur_coord[i];
        done[i] = heap_nodes[i].converter -> done[i];
        logprob_so_far[i] = heap_nodes[i].converter -> logprob_so_far[i];
        bound_togo[i] = heap_nodes[i].converter -> bound_togo[i];
        max_gumbel[i] =heap_nodes[i].converter -> max_gumbel[i];
        is_t_opt[i] = heap_nodes[i].converter -> t_opt[i];
        to_remove.insert(heap_nodes[i].converter);
    }

    vector<torch::Tensor> tensors {ids, t, next_actions, not_visited, prefix, lengths, cur_coord, done,
                                     logprob_so_far, bound_togo, max_gumbel, is_t_opt};

    pair<vector<torch::Tensor>, set<NodeConverter*>> to_return (tensors, to_remove);
    return to_return;
}



class BatchedHeaps{
    private:
        int batch_size_ = 100;
        int graph_size_ = 20;
    public:
        vector<Heap> heaps_;
        BatchedHeaps(NodeConverter &root_nodes, int batch_size, int graph_size){
            batch_size_ = batch_size;
            graph_size_ = graph_size;
            for (int sample_idx=0; sample_idx < batch_size_; sample_idx++){
                //auto new_node = make_shared<HeapNode>(root_nodes[sample_idx]);
                //heaps_.push_back(Heap(new_node));

                heaps_.push_back(Heap(HeapNode(root_nodes[sample_idx])));
            }
        };

        py::list pop_batch(){
            vector<HeapNode> heap_nodes;
            vector<NodeConverter*> converters_to_remove;
            py::list trajectories;
            py::list to_return;

            for (int sample_idx=0; sample_idx < batch_size_; sample_idx++){
                bool to_repop = true; //!heaps_[sample_idx].is_empty();
                while (to_repop){
                    auto node(heaps_[sample_idx].pop());
                    to_repop = node.done;
                    if(to_repop){
                        Trajectory traj(node);
                        trajectories.append(traj);
                        }
                     else{
                        heap_nodes.push_back(node);
                     }
                }
            }
            pair<vector<torch::Tensor>, set<NodeConverter*>> new_node(stack_heap_nodes_vec(heap_nodes, batch_size_, graph_size_));
            //clearVectorContents(heap_nodes);
            to_return.append(new_node);
            to_return.append(trajectories);
            return to_return;
        }

        void push_batch(NodeConverter &node_conv, torch::Tensor ignore_list){
            for (int sample_idx=0; sample_idx < batch_size_; sample_idx++){
                if (!ignore_list[sample_idx].item<bool>()){
                    //new_node = new HeapNode(heap_nodes[sample_idx]);
                    heaps_[sample_idx].push(HeapNode(node_conv[sample_idx]));
                    //heaps_[sample_idx].push(make_shared<HeapNode>(heap_nodes[sample_idx]));
                    //heaps_[sample_idx].push(new HeapNode(heap_nodes[sample_idx]));
                    }
            }
        }
        void push_batch(NodeConverter &node_conv){
            for (int sample_idx=0; sample_idx < batch_size_; sample_idx++){
                heaps_[sample_idx].push(HeapNode(node_conv[sample_idx]));
            }
        }
        int size(){
            return heaps_.size();
        }
};

std::unique_ptr<NodeConverter> convert_python_to_cpp(torch::Tensor _priority, torch::Tensor _t_opt, torch::Tensor _done, torch::Tensor _ids,
                  torch::Tensor _t, torch::Tensor _next_actions, torch::Tensor _not_visited, torch::Tensor _prefix,
                  torch::Tensor _lengths, torch::Tensor _cur_coord, torch::Tensor _logprob_so_far, torch::Tensor _bound_togo,
                  torch::Tensor _max_gumbel, py::object _python_object) {

    return std::unique_ptr<NodeConverter>(new NodeConverter(_priority,_t_opt,_done,_ids,_t,_next_actions,_not_visited,
                    _prefix,_lengths,_cur_coord,_logprob_so_far,_bound_togo,_max_gumbel,_python_object));
 }

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stack_heap_nodes_vec", &stack_heap_nodes_vec);
  m.def("convert_python_to_cpp", &convert_python_to_cpp);
  py::class_<BatchedHeaps>(m, "BatchedHeaps") // returns BatchedHeapNode
      .def(py::init<NodeConverter&, int, int>(), py::keep_alive<1, 2>()) // py::return_value_policy::take_ownership
      //.def(py::init<py::object>())
      .def("pop_batch", &BatchedHeaps::pop_batch)
      .def("push_batch",(void (BatchedHeaps::*)(NodeConverter&)) &BatchedHeaps::push_batch)
      .def("push_batch",(void (BatchedHeaps::*)(NodeConverter&, torch::Tensor)) &BatchedHeaps::push_batch)


      .def("size", &BatchedHeaps::size)
      .def_readonly("heaps_", &BatchedHeaps::heaps_);

  py::class_<HeapNode>(m, "HeapNode")
      //.def(py::init<double,double,bool,double>())
      .def(py::init<pair<py::list, NodeConverter*>>())
      .def_readwrite("python_node_object", &HeapNode::python_node_object)
      .def_readwrite("priority", &HeapNode::priority)
      .def_readwrite("done", &HeapNode::done)
      .def_readwrite("idx_original", &HeapNode::idx_original)
      .def_readwrite("t_opt", &HeapNode::t_opt);


  py::class_<Trajectory>(m, "Trajectory")
      //.def(py::init<HeapNode>())
      .def_readonly("idx", &Trajectory::idx)
      .def_readonly("actions", &Trajectory::actions)
      .def_readonly("cost", &Trajectory::cost)
      .def_readonly("objective", &Trajectory::objective);
  /*
  py::class_<Heap>(m, "Heap")
      .def(py::init<HeapNode*>())
      //.def("get_elements_", &Heap::get_elements_)
      .def("push", &Heap::push)
      .def("pop", &Heap::pop) //
      .def("is_empty", &Heap::is_empty); //pybind11::return_value_policy::move
  */
  py::class_<NodeConverter>(m, "NodeConverter")
      .def(py::init<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,
       torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor, py::object>())
      .def_readwrite("priority", &NodeConverter::priority)
      .def_readwrite("t_opt", &NodeConverter::t_opt)
      .def_readwrite("done", &NodeConverter::done)
      .def_readwrite("ids", &NodeConverter::ids)
      .def_readwrite("t", &NodeConverter::t)
      .def_readwrite("next_actions", &NodeConverter::next_actions)
      .def_readwrite("not_visited", &NodeConverter::not_visited)
      .def_readwrite("prefix", &NodeConverter::prefix)
      .def_readwrite("lengths", &NodeConverter::lengths)
      .def_readwrite("cur_coord", &NodeConverter::cur_coord)
      .def_readwrite("logprob_so_far", &NodeConverter::logprob_so_far)
      .def_readwrite("bound_togo", &NodeConverter::bound_togo)
      .def_readwrite("max_gumbel", &NodeConverter::max_gumbel);

}

