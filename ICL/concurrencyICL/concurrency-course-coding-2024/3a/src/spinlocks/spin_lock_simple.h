#ifndef SPIN_LOCK_SIMPLE_H
#define SPIN_LOCK_SIMPLE_H

#include <atomic>
#include <string>

class SpinLockSimple {
 public:
  SpinLockSimple() {}

  void Lock() {
    while (lock_bit_.exchange(true)) {
      // Try again - i.e. spin
    }
  }

  void Unlock() { lock_bit_.store(false); }

  [[nodiscard]] static std::string GetName() { return "Simple"; }

 private:
  std::atomic<bool> lock_bit_;
};

#endif  // SPIN_LOCK_SIMPLE_H
