#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

class Logger {
 public:
  explicit Logger(size_t max_size) : max_size_(max_size) {}

  void LogMessage(const std::string& message) {
    mutex_.lock();
    if (log_contents_.size() + message.size() > max_size_) {
      // Deadlock will occur due to failure to release mutex.
      return;
    }
    log_contents_ += message;
    mutex_.unlock();
  }

  std::string GetLogContents() { return log_contents_; }

 private:
  std::string log_contents_;
  std::mutex mutex_;
  const size_t max_size_;
};

static void ThreadBody(size_t nice_id,
                       const std::vector<std::string>& favourite_food,
                       Logger& logger) {
  std::stringstream stream;
  stream << "Hello, I am thread " << std::this_thread::get_id()
         << ", but you can call me " << nice_id << ", and my favourite food is "
         << favourite_food[nice_id] << "\n";
  logger.LogMessage(stream.str());
}

int main() {
  Logger logger(200);
  std::vector<std::string> foods = {
      "Ice cream", "Pizza", "Bread",  "Sushi",
      "Snake",     "Pasta", "Apples", "Soy sauce-braised frogs' legs"};
  std::vector<std::thread> threads;
  for (int i = 0; i < 8; i++) {
    threads.push_back(
        std::thread(ThreadBody, i, std::ref(foods), std::ref(logger)));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::cout << logger.GetLogContents() << "\n";
  std::cout.flush();
}
