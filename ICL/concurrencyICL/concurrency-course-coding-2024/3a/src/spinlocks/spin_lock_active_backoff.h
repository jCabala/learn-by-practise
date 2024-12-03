#ifndef SPIN_LOCK_ACTIVE_BACKOFF_H
#define SPIN_LOCK_ACTIVE_BACKOFF_H

#include <atomic>
#include <string>

class SpinLockActiveBackoff {
 public:
  SpinLockActiveBackoff() {}

  void Lock() {
    while (lock_bit_.exchange(true)) {
      // Lock was not free, so locally spin until it is observed to be free
      do {
        // Actively do some redundant work in the hope that we get out of sync
        // with other threads contending for the lock!
        for (volatile int i = 0; i < 100; i = i + 1) {
          // Do nothing
        }
        // Keep locally spinning until the lock is available
      } while (lock_bit_.load());
      // The lock has been seen to be available - try to get it!
    }
  }

  void Unlock() { lock_bit_.store(false); }

  [[nodiscard]] static std::string GetName() { return "Active backoff"; }

 private:
  std::atomic<bool> lock_bit_;
};

#endif  // SPIN_LOCK_ACTIVE_BACKOFF_H
