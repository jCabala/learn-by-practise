#ifndef SPIN_LOCK_PASSIVE_BACKOFF_H
#define SPIN_LOCK_PASSIVE_BACKOFF_H

#include <emmintrin.h>

#include <atomic>
#include <string>

class SpinLockPassiveBackoff {
 public:
  SpinLockPassiveBackoff() {}

  void Lock() {
    while (lock_bit_.exchange(true)) {
      // Lock was not free, so locally spin until it is observed to be free
      do {
        // Passively back off in the hope that we get out of sync
        // with other threads contending for the lock!
        for (int i = 0; i < 4; i++) {
          // This instruction will cause a pause at lower energy consumption.
          _mm_pause();
        }
        // Keep locally spinning until the lock is available
      } while (lock_bit_.load());
      // The lock has been seen to be available - try to get it!
    }
  }

  void Unlock() { lock_bit_.store(false); }

  [[nodiscard]] static std::string GetName() { return "Passive backoff"; }

 private:
  std::atomic<bool> lock_bit_;
};

#endif  // SPIN_LOCK_PASSIVE_BACKOFF_H
