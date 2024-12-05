#include <atomic>
#include <thread>
#include <iostream>
#include <vector>

class CLHQueueLock {
private:
    struct Node {
        std::atomic<bool> locked;
        Node() : locked(false) {}
    };

    std::atomic<Node*> tail;
    static thread_local Node* myNode;
    static thread_local Node* myPred;

public:
    CLHQueueLock() {
        tail.store(new Node());
    }

    ~CLHQueueLock() {
        delete tail.load();
    }

    void lock() {
        myNode = new Node();
        myNode->locked.store(true);
        Node* prevNode = tail.exchange(myNode);
        myPred = prevNode;

        while (prevNode->locked.load()) {
            // Spin
        }
    }

    void unlock() {
        myNode->locked.store(false);
        myNode = myPred;
    }
};

thread_local CLHQueueLock::Node* CLHQueueLock::myNode = nullptr;
thread_local CLHQueueLock::Node* CLHQueueLock::myPred = nullptr;

void threadTask(CLHQueueLock& lock, int threadID) {
    lock.lock();
    std::cout << "Thread " << threadID << " acquired the lock.\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
    std::cout << "Thread " << threadID << " released the lock.\n";
    lock.unlock();
}

int main() {
    CLHQueueLock lock;
    const int numThreads = 5;

    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&lock, i]() { threadTask(lock, i); });
    }

    for (auto& t : threads) {
        t.join();
    }

    return 0;
}