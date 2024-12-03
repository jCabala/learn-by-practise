#include <iostream>
#include <thread>

#include "spinlocks/spin_lock_simple.h"

static int x = 0;
static int y = 0;

static void T1(SpinLockSimple& spin_lock) {
  spin_lock.Lock();
  // Critical section
  x = 42;
  y = 43;
  spin_lock.Unlock();
}

static void T2(SpinLockSimple& spin_lock) {
  int t1;
  int t2;
  spin_lock.Lock();
  t1 = x;
  t2 = y;
  spin_lock.Unlock();
  std::cout << t1 << " " << t2 << std::endl;
}

int main() {
  SpinLockSimple spin_lock;
  std::thread t1(T1, std::ref(spin_lock));
  std::thread t2(T2, std::ref(spin_lock));
  t1.join();
  t2.join();
}
