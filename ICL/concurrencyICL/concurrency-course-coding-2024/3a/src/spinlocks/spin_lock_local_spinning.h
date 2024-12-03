#ifndef SPIN_LOCK_LOCAL_SPINNING_H
#define SPIN_LOCK_LOCAL_SPINNING_H

#include <atomic>
#include <string>

class SpinLockLocalSpinning {
 public:
  SpinLockLocalSpinning() {}

  void Lock() {
    while (lock_bit_.exchange(true)) {
      // Lock was not free, so locally spin until it is observed to be free
      while (lock_bit_.load()) {
        // Keep locally spinning until the lock is available
      }
      // The lock has been seen to be available - try to get it!
    }
  }

  void Unlock() { lock_bit_.store(false); }

  [[nodiscard]] static std::string GetName() { return "Local spinning"; }

 private:
  std::atomic<bool> lock_bit_;
};

#endif  // SPIN_LOCK_LOCAL_SPINNING_H
