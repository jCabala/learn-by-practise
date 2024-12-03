#ifndef SPIN_LOCK_SIMPLE_RELAXED_H
#define SPIN_LOCK_SIMPLE_RELAXED_H

#include <atomic>
#include <string>

class SpinLockSimpleRelaxed {
 public:
  SpinLockSimpleRelaxed() : lock_bit_(false) {}

  void Lock() {
    while (lock_bit_.exchange(true, std::memory_order_relaxed)) {
    }
  }

  void Unlock() { lock_bit_.store(false, std::memory_order_relaxed); }

  [[nodiscard]] static std::string GetName() { return "Simple relaxed"; }

 private:
  std::atomic<bool> lock_bit_;
};

#endif  // SPIN_LOCK_SIMPLE_RELAXED_H
