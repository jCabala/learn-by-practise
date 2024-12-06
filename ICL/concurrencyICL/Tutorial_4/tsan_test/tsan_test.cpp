#include <chrono>
#include <thread>
#include <mutex>
#include <iostream>

void ThreadBody(std::mutex& mutex, int& counter) {
    for (int i = 0; i < 100; i++) {
        if (i != 50) {
            mutex.lock();
        } else {
            //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }

        counter++;
        if (i != 50) {
            mutex.unlock();
        }
    }
}

int main() {
    std::mutex mutex;
    int counter = 0;
    auto t = std::thread(ThreadBody, std::ref(mutex), std::ref(counter));
    //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    mutex.lock();
    std::cout << "Counter: " << counter << std::endl;
    mutex.unlock();
    t.join();
    return 0;
}