#include <chrono>
#include <iostream>
#include <thread>

static void Foo(int& x) {
  std::cout << "Waiting\n";
  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout << "Waking up thread 2\n";
  x = 1;
}

static void Bar(int& x) {
  while (x == 0) {
    // Do nothing
  }
  std::cout << "Thread 2 got woken up\n";
}

int main() {
  int x = 0;
  std::thread t1(Foo, std::ref(x));
  std::thread t2(Bar, std::ref(x));
  t1.join();
  t2.join();
  return 0;
}
