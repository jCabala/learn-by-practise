#ifndef SPIN_LOCK_ALOCK_PADDED_H
#define SPIN_LOCK_ALOCK_PADDED_H

#include <array>
#include <atomic>
#include <string>

class SpinLockALockPadded {
 public:
  SpinLockALockPadded() : tail_(0), flag_() {
    flag_[0].flag_bit.store(true);
    for (size_t i = 1; i < kCapacity; i++) {
      flag_[i].flag_bit.store(false);
    }
  }

  void Lock() {
    my_slot_index_ = tail_.fetch_add(1) % kCapacity;
    while (!flag_[my_slot_index_].flag_bit.load()) {
      // Spin
    }
  }

  void Unlock() {
    flag_[my_slot_index_].flag_bit.store(false);
    flag_[(my_slot_index_ + 1) % kCapacity].flag_bit.store(true);
  }

  [[nodiscard]] static std::string GetName() { return "ALock padded"; }

 private:
  struct padded_flag_entry {
    std::atomic<bool> flag_bit;
  } __attribute__((aligned(64)));  // I have used 64 because this is the size of
                                   // a cache line on my system.

  static const size_t kCapacity = 16;
  static thread_local size_t my_slot_index_;
  std::atomic<size_t> tail_;
  std::array<padded_flag_entry, kCapacity> flag_;
};

#endif  // SPIN_LOCK_ALOCK_PADDED_H
