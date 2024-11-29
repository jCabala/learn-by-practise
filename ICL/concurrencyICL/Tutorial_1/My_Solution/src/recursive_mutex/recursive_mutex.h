#ifndef RECURSIVE_MUTEX_BASE_H
#define RECURSIVE_MUTEX_BASE_H
#include <cassert>
#include <condition_variable>
#include <mutex>
#include <thread>

class RecursiveMutex {
 public:
  RecursiveMutex() = default;
  void Lock() {
    std::unique_lock<std::mutex> condition_lock(mutex_);
    condition_.wait(condition_lock, [this]() {
       return lockCount_ == 0 || holder_ == std::this_thread::get_id();
    });
    holder_ = std::this_thread::get_id();
    lockCount_++;
  }

  void Unlock() {
    assert(std::this_thread::get_id() == holder_);
    lockCount_--;
    if (lockCount_ == 0) condition_.notify_one();
  }

 private:
  int lockCount_ = 0;
  std::thread::id holder_;
  std::mutex mutex_;
  std::condition_variable condition_;
};

#endif  // RECURSIVE_MUTEX_BASE_H
