#include <iostream>
#include <mutex>
#include <thread>


// int foo(int x) {
//
// }

static int bar(int& x) {
  int temp = x;
  x = 42;
  x++;
  x = 52;
  int* nasty_pointer = &x;
  nasty_pointer++;
  *nasty_pointer = 202;
  return temp;
}

// int baz(int& x) {
//
//
// }


int main() {
  int a = 0;
  int b = 1;
  std::cout << bar(a) << std::endl;
  std::cout << a << std::endl;
  std::cout << b << std::endl;
}
