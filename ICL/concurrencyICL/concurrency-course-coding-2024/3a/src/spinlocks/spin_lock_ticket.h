#ifndef SPIN_LOCK_TICKET_H
#define SPIN_LOCK_TICKET_H

#include <emmintrin.h>

#include <atomic>
#include <string>

class SpinLockTicket {
 public:
  SpinLockTicket() {}

  void Lock() {
    int my_ticket = next_ticket_.fetch_add(1);
    while (my_ticket != now_serving_.load()) {
      // Passively do nothing - wait
      _mm_pause();
    }
    // When we here, our ticket is the ticket being served - i.e., we have the
    // lock!
  }

  void Unlock() {
    // I am done, so the next ticket can be served
    now_serving_.store(now_serving_.load() + 1);
  }

  [[nodiscard]] static std::string GetName() { return "Ticket"; }

 private:
  // The next ticket to be given out
  std::atomic<int> next_ticket_;
  // The value of the ticket currently being served
  std::atomic<int> now_serving_;
};

#endif  // SPIN_LOCK_TICKET_H
