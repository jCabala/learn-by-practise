#ifndef SHARED_MUTEX_FAIR_H
#define SHARED_MUTEX_FAIR_H

#include <cassert>
#include <condition_variable>
#include <mutex>

#include "src/shared_mutexes/shared_mutex_base.h"

class SharedMutexFair : public SharedMutexBase {
 public:
  void Lock() final {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]() -> bool { return !writer_; });
    writer_ = true;
    condition_.wait(
        lock, [this]() -> bool { return read_acquires_ == read_releases_; });
  }

  void Unlock() final {
    std::unique_lock<std::mutex> lock(mutex_);
    writer_ = false;
  }

  void LockShared() final {
    std::unique_lock<std::mutex> lock(mutex_);
    read_acquires_++;
    condition_.wait(lock, [this]() {
      return !writer_;
    });
  }

  void UnlockShared() final {
    std::unique_lock<std::mutex> lock(mutex_);
    if (read_acquires_ == read_releases_) { condition_.notify_all(); }
    read_releases_++;
  }

 private:
  int read_acquires_ = 0, read_releases_ = 0;
  bool writer_ = false;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::mutex readLock_, writeLock_;
};

#endif  // SHARED_MUTEX_FAIR_H
