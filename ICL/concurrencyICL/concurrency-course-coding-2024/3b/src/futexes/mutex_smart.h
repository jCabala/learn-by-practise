#ifndef MUTEX_SMART_H
#define MUTEX_SMART_H

#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <string>

class MutexSmart {
 public:
  MutexSmart() : state_(kFree) {}

  void Lock() {
    // Try to get the lock on the fast path, by attempting to change the futex
    // word from kFree to kLockedNoWaiters
    int old_value = cmpxchg(kFree, kLockedNoWaiters);
    if (old_value == kFree) {
      // We got the mutex on the fast path -- job done!
      return;
    }
    // Slow path -- we didn't manage to directly get the mutex!
    do {
      // Prepare to go to sleep, by changing the futex word to kLockedWaiters
      // This important because *we* will be a waiter, so we need to record that
      // there are waiters!

      // The following will either:
      // - succeed - good; we have recorded that there are waiters
      // - fail because the state is already kLockedWaiters - fine
      // - fail because the state has suddenly become kFree - also fine, the
      //   upcoming futex_wait will be a no-op and we can try to get the mutex
      //   again
      if (old_value == kLockedWaiters ||  // We definitely need to go to sleep
                                          // If we see that the mutex is free,
                                          // don't call futex_wait
          cmpxchg(kLockedNoWaiters, kLockedWaiters) != kFree) {
        // Call futex_wait
        syscall(SYS_futex, reinterpret_cast<int*>(&state_), FUTEX_WAIT,
                kLockedWaiters, nullptr, nullptr, 0);
      }

      // We have been woken up (or we didn't actually go to sleep) - let's try
      // to get the mutex again. Pessimistically assume there are waiters (as
      // we were a waiter (or nearly!)
      old_value = cmpxchg(kFree, kLockedWaiters);
    } while (old_value != kFree);
    // We got the mutex on the slow path
  }

  void Unlock() {
    if (state_.exchange(kFree) == kLockedWaiters) {
      // There might be waiters, so wake one of them up!
      syscall(SYS_futex, reinterpret_cast<int*>(&state_), FUTEX_WAKE,
              1,  // Wake up one thread
              nullptr, nullptr, 0);
    }
  }

  [[nodiscard]] static std::string GetName() { return "Smart"; }

 private:
  // The lock is available
  const int kFree = 0;

  // The lock is not available, and whoever locked it had no reason to believe
  // there are waiters (though there might actually be waiters!)
  const int kLockedNoWaiters = 1;

  // The lock is not available, and whoever locked it had reason to believe
  // that there might be waiters (though actually there might be none!)
  const int kLockedWaiters = 2;

  std::atomic<int> state_;

  int cmpxchg(int expected, int desired) {
    state_.compare_exchange_strong(expected, desired);
    return expected;
  }
};

#endif  // MUTEX_SMART_H
