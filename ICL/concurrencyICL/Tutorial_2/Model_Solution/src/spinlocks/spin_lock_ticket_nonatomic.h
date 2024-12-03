#ifndef SPIN_LOCK_TICKET_NON_ATOMIC_H
#define SPIN_LOCK_TICKET_NON_ATOMIC_H

#include <emmintrin.h>

#include <atomic>
#include <string>

class SpinLockTicketNonAtomic {
 public:
  SpinLockTicketNonAtomic() : next_ticket_(0), now_serving_(0) {}

  void Lock() {
    const auto ticket = next_ticket_.fetch_add(1);
    // Incorrect: because now_serving_ is not atomic, a load by one thread
    // cannot synchronise with a store by another thread. The compiler is
    // free to hold now_serving_ in a register, so that the loop will
    // become infinite.
    while (now_serving_ != ticket) {
      _mm_pause();
    }
  }

  void Unlock() {
    // Similarly incorrect.
    now_serving_ = now_serving_ + 1;
  }

  [[nodiscard]] static std::string GetName() { return "Ticket nonatomic"; }

 private:
  std::atomic<size_t> next_ticket_;
  size_t now_serving_;  // Incorrect: needs to be atomic
};

#endif  // SPIN_LOCK_TICKET_NON_ATOMIC_H
