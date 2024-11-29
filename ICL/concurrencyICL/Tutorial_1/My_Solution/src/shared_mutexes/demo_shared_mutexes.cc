#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "shared_mutex_simple.h"
#include "shared_mutex_fair.h"
#include "shared_mutex_native.h"
#include "shared_mutex_stupid.h"

template <typename T>
void testMutex(T mutex, const std::string& name) {
  std::cout << name << std::endl;
  auto begin_time = std::chrono::high_resolution_clock::now();
  int val = 0;
  std::vector<std::thread> writers;
  std::vector<std::thread> readers;

  for (int i = 1; i <= 500; i++) {
    for (int _ = 0; _ < 10; _++) {
      readers.emplace_back([&val, &mutex]() {
        mutex.LockShared();
        std::cout << val << " ";
        mutex.UnlockShared();
      });
    }

    writers.emplace_back([i, &val, &mutex]() {
      mutex.LockShared();
      val = 100 * i;
      mutex.UnlockShared();
    });
  }

  for (auto &thread : writers) thread.join();
  for (auto &thread : readers) thread.join();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = end_time - begin_time;
  auto millis =
    std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
  std::cout << std::endl << "Duration: " << millis << " ms" << std::endl << std::endl;
}

int main() {
  testMutex(SharedMutexSimple(), "Simple");
  testMutex(SharedMutexNative(), "Native");
  testMutex(SharedMutexFair(), "Fair");
  testMutex(SharedMutexStupid(),"Stupid");

  return 0;
}
