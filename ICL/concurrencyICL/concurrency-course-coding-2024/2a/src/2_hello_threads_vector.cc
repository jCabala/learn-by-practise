#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

static void ThreadBody() {
  std::cout << "Hello, I am thread " << std::this_thread::get_id() << "\n";
}

int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; i++) {
    threads.push_back(std::thread(ThreadBody));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}
