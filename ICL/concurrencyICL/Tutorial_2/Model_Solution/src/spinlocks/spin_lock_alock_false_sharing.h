#ifndef SPIN_LOCK_ALOCK_FALSE_SHARING_H
#define SPIN_LOCK_ALOCK_FALSE_SHARING_H

#include <array>
#include <atomic>
#include <string>

class SpinLockALockFalseSharing {
 public:
  SpinLockALockFalseSharing() : tail_(0), flag_() {
    flag_[0].store(true);
    for (size_t i = 1; i < kCapacity; i++) {
      flag_[i].store(false);
    }
  }

  void Lock() {
    my_slot_index_ = tail_.fetch_add(1) % kCapacity;
    while (!flag_[my_slot_index_].load()) {
      // Spin
    }
  }

  void Unlock() {
    flag_[my_slot_index_].store(false);
    flag_[(my_slot_index_ + 1) % kCapacity].store(true);
  }

  [[nodiscard]] static std::string GetName() { return "ALock false sharing"; }

 private:
  static const size_t kCapacity = 16;
  static thread_local size_t my_slot_index_;
  std::atomic<size_t> tail_;
  std::array<std::atomic<bool>, kCapacity> flag_;
};

#endif  // SPIN_LOCK_ALOCK_FALSE_SHARING_H
