#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class LockedQueue {
 public:
  LockedQueue(size_t capacity) : count_(0), head_(0), tail_(0) {
    elements_.resize(capacity);
  }

  void Enq(int element) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Block until the queue becomes non-full
    not_full_.wait(lock, [this]() {
      return count_ < elements_.size();
    });

    // There is space for us to add an element
    elements_[tail_] = element;
    count_++;
    tail_ = (tail_ + 1) % elements_.size();
    not_empty_.notify_one();
  }

  int Deq() {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait for something to be added
    not_empty_.wait(lock, [this]() {
      return count_ > 0;
    });

    // There is something to be removed from the queue
    int result = elements_[head_];
    count_--;
    head_ = (head_ + 1) % elements_.size();
    not_full_.notify_one();
    return result;
  }

 private:
  std::vector<int> elements_;
  size_t count_;
  size_t head_;
  size_t tail_;
  std::mutex mutex_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
};

int main() {
  const int N_ELEMENTS = 1 << 24;
  const int N_CONSUMERS = 8;
  const int ELEMS_PER_CONSUMER = N_ELEMENTS / N_CONSUMERS;
  LockedQueue producer_to_consumers(256);
  LockedQueue consumers_to_producer(N_CONSUMERS);

  int final_result = 0;

  std::thread producer([&consumers_to_producer, &final_result,
                        &producer_to_consumers]() -> void {
    for (int i = 0; i < N_ELEMENTS; i++) {
      producer_to_consumers.Enq(1);
    }
    for (int i = 0; i < N_CONSUMERS; i++) {
      final_result += consumers_to_producer.Deq();
    }
  });

  std::vector<std::thread> consumers;
  for (int i = 0; i < N_CONSUMERS; i++) {
    consumers.push_back(
        std::thread([&consumers_to_producer, &producer_to_consumers]() -> void {
          int my_sum = 0;
          for (int j = 0; j < ELEMS_PER_CONSUMER; j++) {
            my_sum += producer_to_consumers.Deq();
          }
          consumers_to_producer.Enq(my_sum);
        }));
  }

  producer.join();
  for (auto& consumer : consumers) {
    consumer.join();
  }

  std::cout << final_result << std::endl;
}
