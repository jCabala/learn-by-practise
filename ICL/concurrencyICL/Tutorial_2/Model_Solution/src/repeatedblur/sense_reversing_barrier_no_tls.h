#ifndef SENSE_REVERSING_BARRIER_NO_TLS_H
#define SENSE_REVERSING_BARRIER_NO_TLS_H

#include <atomic>

class SenseReversingBarrierNoTls {
 public:
  SenseReversingBarrierNoTls(size_t num_participants)
      : sense_(false),
        count_(num_participants),
        num_participants_(num_participants) {}

  void Await() {
    bool my_sense = !sense_.load();

    if (count_.fetch_sub(1) == 1) {
      count_.store(num_participants_);
      sense_.store(my_sense);
    } else {
      while (sense_.load() != my_sense) {
        // Spin
      }
    }
  }

 private:
  std::atomic<bool> sense_;
  std::atomic<size_t> count_;
  std::size_t num_participants_;
};

#endif  // SENSE_REVERSING_BARRIER_NO_TLS_H
