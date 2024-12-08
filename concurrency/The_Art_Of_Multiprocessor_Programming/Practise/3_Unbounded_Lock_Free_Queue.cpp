#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <mutex>

template <typename T>
class ULFQueue {
private:
    struct Node {
        T val;
        std::atomic<Node*> next;
        Node(T x) : val(x), next(nullptr) {}
    };
    std::atomic<Node*> head;
    std::atomic<Node*> tail;
    std::vector<Node*> garbage;

public:
    ULFQueue() {
        head = tail = new Node(T()); // Head is a dummy node
    }

    ~ULFQueue() {
        Node* curr = head.load();
        while (curr) {
            Node* next = curr->next.load();
            delete curr;
            curr = next;
        }
    }

    void enqueue(T x) {
        Node* node = new Node(x);
        Node* last = nullptr;
        Node* next = nullptr;
        while (true) { // Keep trying until enqueued
            last = tail.load();
            next = last->next.load();
            if (last == tail.load()) {
                if (next == nullptr) { // If tail is pointing to last node try to enqueue 
                    if (last->next.compare_exchange_strong(next, node)) {
                        break; // Enqueued successfully
                    }
                } else { // If successor exists, help other threads advance tail
                    tail.compare_exchange_strong(last, next);
                }
            }
        }
        tail.compare_exchange_strong(last, node); // If failed another thread has advanced tail
    }

    bool dequeue(T& x) {
        Node* first = nullptr;
        Node* last = nullptr;
        Node* next = nullptr;
        while (true) {
            first = head.load();
            last = tail.load();
            next = first->next.load();
            if (first == head.load()) {
                if (first == last) { // We need to take care of tail pointer to not point to a node that is being deleted
                    if (next == nullptr) { // If queue is empty
                        return false;
                    }
                    tail.compare_exchange_strong(last, next);
                } else {
                    x = next->val; // Dequeue the value, remember first is a dummy node
                    if (head.compare_exchange_strong(first, next)) { // Dequeue successful
                        break;
                    }
                    // If failed another thread has advanced head so try again
                }
            }
        }
        delete first;
        return true;
    }
};

int main() {
    ULFQueue<int> queue;

    const int num_producers = 4;
    const int num_consumers = 4;
    const int num_items = 50;

    std::atomic<int> produced_count{0};
    std::atomic<int> consumed_count{0};
    std::mutex print_mutex;

    // Producer threads
    std::vector<std::thread> producers;
    for (int i = 0; i < num_producers; ++i) {
        producers.emplace_back([&queue, &produced_count, num_items, i]() {
            for (int j = 0; j < num_items; ++j) {
                int item = i * num_items + j;
                queue.enqueue(item);
                ++produced_count;
                std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 10));
            }
        });
    }

    // Consumer threads
    std::vector<std::thread> consumers;
    for (int i = 0; i < num_consumers; ++i) {
        consumers.emplace_back([&queue, &consumed_count, &print_mutex]() {
            int item;
            while (consumed_count.load() < num_producers * num_items) {
                if (queue.dequeue(item)) {
                    ++consumed_count;
                    print_mutex.lock();
                    std::cout << "Consumed: " << item << "\n";
                    print_mutex.unlock();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 10));
            }
        });
    }

    // Join threads
    for (auto& producer : producers) {
        producer.join();
    }
    for (auto& consumer : consumers) {
        consumer.join();
    }

    std::cout << "Produced: " << produced_count.load() << ", Consumed: " << consumed_count.load() << "\n";

    return 0;
}
