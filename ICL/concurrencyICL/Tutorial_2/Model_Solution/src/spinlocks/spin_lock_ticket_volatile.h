#ifndef SPIN_LOCK_TICKET_VOLATILE_H
#define SPIN_LOCK_TICKET_VOLATILE_H

#include <emmintrin.h>

#include <atomic>
#include <string>

class SpinLockTicketVolatile {
 public:
  SpinLockTicketVolatile() : next_ticket_(0), now_serving_(0) {}

  void Lock() {
    const auto ticket = next_ticket_.fetch_add(1);
    // Incorrect: because now_serving_ is not atomic, a load by one thread
    // cannot synchronise with a store by another thread. The use of volatile
    // might mean that the lock works in practice on some platforms, but there
    // are no guarantees.
    while (now_serving_ != ticket) {
      _mm_pause();
    }
  }

  void Unlock() {
    // Similarly incorrect.
    now_serving_ = now_serving_ + 1;
  }

  [[nodiscard]] static std::string GetName() { return "Ticket volatile"; }

 private:
  std::atomic<size_t> next_ticket_;
  volatile size_t now_serving_;  // Incorrect: needs to be atomic, not volatile
};

#endif  // SPIN_LOCK_TICKET_VOLATILE_H
