#ifndef SHARED_MUTEX_STUPID_H
#define SHARED_MUTEX_STUPID_H

#include <cassert>
#include <condition_variable>
#include <mutex>

#include "src/shared_mutexes/shared_mutex_base.h"

class SharedMutexStupid : public SharedMutexBase {
 public:
  void Lock() final {
    mutex_.lock();
  }

  void Unlock() final {
    mutex_.unlock();
  }

  void LockShared() final {
    mutex_.lock();
  }

  void UnlockShared() final {
    mutex_.unlock();
  }

 private:
  std::mutex mutex_;
};

#endif  // SHARED_MUTEX_STUPID_H
