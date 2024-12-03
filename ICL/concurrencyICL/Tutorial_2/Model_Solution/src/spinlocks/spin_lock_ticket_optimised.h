#ifndef SPIN_LOCK_TICKET_OPTIMISED_H
#define SPIN_LOCK_TICKET_OPTIMISED_H

#include <emmintrin.h>

#include <atomic>
#include <string>

class SpinLockTicketOptimised {
 public:
  SpinLockTicketOptimised() : next_ticket_(0), now_serving_(0) {}

  void Lock() {
    const auto ticket = next_ticket_.fetch_add(1);
    while (true) {
      size_t ns = now_serving_.load();
      if (ns == ticket) {
        break;
      }
      // Back off for a number of iterations proportional to the "distance"
      // between my ticket and the currently-served ticket. The multiplier 4 has
      // been chosen arbitrarily as a multiplier; it's hard to pick a good
      // one-size-fits-all value.
      for (size_t i = 0; i < ticket - ns; i++) {
        _mm_pause();
      }
    }
  }

  void Unlock() { now_serving_.store(now_serving_.load() + 1); }

  [[nodiscard]] static std::string GetName() { return "Ticket optimised"; }

 private:
  std::atomic<size_t> next_ticket_;
  std::atomic<size_t> now_serving_;
};

#endif  // SPIN_LOCK_TICKET_OPTIMISED_H
