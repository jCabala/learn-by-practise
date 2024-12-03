#ifndef SPIN_LOCK_EXPONENTIAL_BACKOFF_H
#define SPIN_LOCK_EXPONENTIAL_BACKOFF_H

#include <emmintrin.h>

#include <atomic>

class SpinLockExponentialBackoff {
 public:
  SpinLockExponentialBackoff() {}

  void Lock() {
    const int kMinBackoff = 4;
    const int kMaxBackoff = 1 << 10;
    while (lock_bit_.exchange(true)) {
      int backoff_period = kMinBackoff;
      // Lock was not free, so locally spin until it is observed to be free
      do {
        // Passively back off in the hope that we get out of sync
        // with other threads contending for the lock!
        for (int i = 0; i < backoff_period; i++) {
          // This instruction will cause a pause at lower energy consumption.
          _mm_pause();
        }
        backoff_period = std::min(2 * backoff_period, kMaxBackoff);
        // Keep locally spinning until the lock is available
      } while (lock_bit_.load());
      // The lock has been seen to be available - try to get it!
    }
  }

  void Unlock() { lock_bit_.store(false); }

  [[nodiscard]] static std::string GetName() { return "Exponential backoff"; }

 private:
  std::atomic<bool> lock_bit_;
};

#endif  // SPIN_LOCK_EXPONENTIAL_BACKOFF_H
