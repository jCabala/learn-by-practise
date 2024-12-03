#include <atomic>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

template <typename T>
size_t Find(const std::vector<T>& data, std::function<bool(T)> predicate) {
  std::atomic<size_t> result(std::numeric_limits<size_t>::max());

  std::function<void(size_t, size_t)> thread_body =
      [&result, &data, &predicate](size_t start, size_t end) -> void {
    for (size_t i = start; i < end; i++) {
      if (predicate(data[i])) {
        result.store(i, std::memory_order_relaxed);
        return;
      }
    }
  };

  // Processes first half of `data`
  std::thread t1(thread_body, 0, data.size() / 2);
  // Processes second half of `data`
  std::thread t2(thread_body, data.size() / 2, data.size());
  t1.join();
  t2.join();
  return result.load(std::memory_order_relaxed);
}

int main() {
  std::vector<int> data;
  for (size_t count = 0; count < 2; count++) {
    for (int i = 0; i < (1 << 24); i++) {
      data.push_back(i);
    }
  }
  size_t result =
      Find<int>(data, [](int item) -> bool { return item == 16000000; });

  if (result != std::numeric_limits<size_t>::max()) {
    std::cout << "Found " << data[result] << " at index " << result
              << std::endl;
  } else {
    std::cout << "Not found" << std::endl;
  }
  return 0;
}
