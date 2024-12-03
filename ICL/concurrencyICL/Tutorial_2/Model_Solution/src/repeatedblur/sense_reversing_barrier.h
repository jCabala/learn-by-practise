#ifndef SENSE_REVERSING_BARRIER_H
#define SENSE_REVERSING_BARRIER_H

#include <atomic>

class SenseReversingBarrier {
 public:
  SenseReversingBarrier(size_t num_participants)
      : sense_(false),
        count_(num_participants),
        num_participants_(num_participants) {}

  void Await() {
    if (count_.fetch_sub(1) == 1) {
      count_.store(num_participants_);
      sense_.store(my_sense_);
    } else {
      while (sense_.load() != my_sense_) {
        // Spin
      }
    }
    my_sense_ = !my_sense_;
  }

 private:
  std::atomic<bool> sense_;
  std::atomic<size_t> count_;
  std::size_t num_participants_;
  static thread_local bool my_sense_;
};

#endif  // SENSE_REVERSING_BARRIER_H
