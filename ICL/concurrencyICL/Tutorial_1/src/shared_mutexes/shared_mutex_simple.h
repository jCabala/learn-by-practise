#ifndef SHARED_MUTEX_SIMPLE_H
#define SHARED_MUTEX_SIMPLE_H

#include <cassert>
#include <condition_variable>
#include <mutex>

#include "src/shared_mutexes/shared_mutex_base.h"

class SharedMutexSimple : public SharedMutexBase {
 public:
  void Lock() final {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() {
      return num_readers_ == 0 && !has_writer_;
    });
    has_writer_ = true;
  }

  void Unlock() final {
      assert(has_writer_);
      has_writer_ = false;
      condition_.notify_all();
  }

  void LockShared() final {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() {
      return !has_writer_;
    });
    num_readers_++;
  }

  void UnlockShared() final {
    assert(num_readers_ > 0);
    num_readers_--;
    if (num_readers_ == 0) condition_.notify_all();
  }

 private:
  int num_readers_ = 0;
  bool has_writer_ = false;
  std::condition_variable condition_;
  std::mutex mutex_;
};

#endif  // SHARED_MUTEX_SIMPLE_H
