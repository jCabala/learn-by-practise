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
    condition_.wait(lock, [this]() {
      return readAcquires_ == readReleases_;
    });
    writer_ = true;
  }

  void Unlock() final {
    writer_ = false;
  }

  void LockShared() final {
    std::unique_lock<std::mutex> lock(mutex_);
    readAcquires_++;
    condition_.wait(lock, [this]() {
      return !writer_;
    });
  }

  void UnlockShared() final {
    readReleases_++;
    if (readAcquires_ == readReleases_) {
      condition_.notify_all();
    }
  }

 private:
  int readAcquires_ = 0, readReleases_ = 0;
  bool writer_ = false;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::mutex readLock_, writeLock_;
};

#endif  // SHARED_MUTEX_FAIR_H
