#include <chrono>
#include <iostream>
#include <thread>

static void ThreadBody1() {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  std::cout << "Hello\n";
}

int main() {
  std::thread t1(ThreadBody1);
  std::thread t2([/* variable captured by the lambda - none here*/](
                     /* parameters to the lambda - none here */) -> void {
    // The body of the lambda - this is what the launched thread
    // will execute.
    std::cout << "World\n";
  });

  t1.join();
  t2.join();
}
