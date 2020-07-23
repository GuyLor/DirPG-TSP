#ifndef UNION_FIND_H__
#define UNION_FIND_H__


#include <iostream>
#include <vector>

using namespace std;


class UnionFind {

public:
  UnionFind(int size);

  void reset();

  //void dump();

  int findRoot(int i);

  void merge(int i, int j);

protected:
  vector<int> parents_;
  vector<int> sizes_;
};


#endif  // UNION_FIND__
