#include <thread>
#include <vector>

#include "container.h"

int main() {
  for (int _ = 0; _ < 100; _++) {
    std::vector<std::thread> threads;
    Container<char> container;

    for (uint i = 0; i < 5; i++) {
      char c = static_cast<char>(static_cast<int>('a') + i);
      threads.emplace_back([c, &container]() {
        container.AddAll({c, c, c});
      });
    }


    for (auto &t : threads) t.join();
    container.Show();
    std::cout << "\n";
  }

  return 0;
}
