#include "union_find.h"

#include <algorithm>

UnionFind::UnionFind(int size) {
  parents_.resize(size);
  sizes_.resize(size);

  this->reset();
}


void UnionFind::reset() {
  for (int i = 0; i < parents_.size(); i++) {
    parents_[i] = i;
    sizes_[i] = 1;
  }
}

/*
void UnionFind::dump() {
  for (int i = 0; i < parents_.size(); i++) {
    cout << findRoot(i) << "(" << parents_[i] << "," << sizes_[i] << ") ";
  }
  cout << endl;
}
*/

int UnionFind::findRoot(int i) {
  int cur = i;
  while (cur != parents_[cur]) {
    cur = parents_[cur];
  }
  int root = cur;

  int tmp;
  while (cur != root) {
    tmp = parents_[cur];
    parents_[cur] = root;
    cur = tmp;
  }

  return root;
}


void UnionFind::merge(int smaller, int bigger) {
  // Assumes `smaller` and `bigger` are in different groups.

  // Swap to make sure smaller is indeed smaller.
  if (sizes_[bigger] < sizes_[smaller]) {
    swap(smaller, bigger);
  }

  int smaller_root = findRoot(smaller);
  int bigger_root = findRoot(bigger);

  parents_[smaller_root] = bigger_root;
  sizes_[bigger_root] += sizes_[smaller_root];
}
