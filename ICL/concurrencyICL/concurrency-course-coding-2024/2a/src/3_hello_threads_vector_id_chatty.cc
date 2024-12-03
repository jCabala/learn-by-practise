#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

static void ThreadBody(size_t nice_id,
                       const std::vector<std::string>& favourite_food) {
  std::cout << "Hello, I am thread " << std::this_thread::get_id()
            << ", but you can call me " << nice_id
            << ", and my favourite food is " << favourite_food[nice_id] << "\n";
}

int main() {
  std::vector<std::string> foods = {
      "Ice cream", "Pizza", "Bread",  "Sushi",
      "Snake",     "Pasta", "Apples", "Soy sauce-braised frogs' legs"};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; i++) {
    threads.push_back(std::thread(ThreadBody, i, foods));
  }
  for (auto& thread : threads) {
    thread.join();
  }
}
