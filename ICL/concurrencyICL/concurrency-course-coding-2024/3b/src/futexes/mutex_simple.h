#ifndef MUTEX_SIMPLE_H
#define MUTEX_SIMPLE_H

#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <string>

class MutexSimple {
 public:
  MutexSimple() : state_(kFree) {}

  void Lock() {
    while (state_.exchange(kLocked) != kFree) {
      // We didn't manage to get the lock straight away, so go to sleep.
      syscall(SYS_futex, reinterpret_cast<int*>(&state_), FUTEX_WAIT, kLocked,
              nullptr, nullptr, 0);
    }
  }

  void Unlock() {
    state_.store(kFree);
    syscall(SYS_futex, reinterpret_cast<int*>(&state_), FUTEX_WAKE,
            1,  // Wake up one thread
            nullptr, nullptr, 0);
  }

  [[nodiscard]] static std::string GetName() { return "Simple"; }

 private:
  const int kFree = 0;
  const int kLocked = 1;
  std::atomic<int> state_;
};

#endif  // MUTEX_SIMPLE_H
