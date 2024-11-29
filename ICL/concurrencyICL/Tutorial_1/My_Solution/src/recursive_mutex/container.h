#ifndef CONTAINER_H
#define CONTAINER_H
#include <iostream>
#include <vector>

#include "recursive_mutex.h"

template <typename T>
class Container {
 public:
  void Add(const T& elem) {
    mutex_.Lock();
    elems_.push_back(elem);
    mutex_.Unlock();
  }

  void AddAll(const std::vector<T>& elems) {
    mutex_.Lock();
    for(auto elem: elems) Add(elem);
    mutex_.Unlock();
  }

  void Show() {
    for (auto elem: elems_) std::cout << elem << " ";
  }

 private:
  std::vector<T> elems_;
  RecursiveMutex mutex_;
};

#endif  // CONTAINER_H
