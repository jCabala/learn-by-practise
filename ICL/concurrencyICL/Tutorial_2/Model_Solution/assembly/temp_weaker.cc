#include <atomic>
#include <string>

class SpinLockActiveBackoffWeakerOrderings {
 public:
  SpinLockActiveBackoffWeakerOrderings() : lock_bit_(false) {}

  void Lock() {
    while (lock_bit_.exchange(true, std::memory_order_acquire)) {
      do {
        for (volatile size_t i = 0; i < 100; i++) {
          // Do nothing
        }
      } while (lock_bit_.load(std::memory_order_relaxed));
    }
  }

  void Unlock() { lock_bit_.store(false, std::memory_order_release); }

  [[nodiscard]] static std::string GetName() {
    return "Active backoff weaker orderings";
  }

 private:
  std::atomic<bool> lock_bit_;
};

int main() {
  SpinLockActiveBackoffWeakerOrderings lock;
  lock.Lock();
  lock.Unlock();
}
