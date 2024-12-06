#pragma once

template <class T>
class Set {
public:
    virtual bool add(T x) = 0;
    virtual bool remove(T x) = 0;
    virtual bool contains(T x) = 0;
    virtual void printName() = 0;

    struct Node {
        T item;
        int key;
        Node* next;
        std::mutex mutex;
        Node(int key, T item) : key(key), item(item), next(nullptr) {}
        Node(int key) : key(key), item(T()), next(nullptr) {}

        void lock() {
            mutex.lock();
        }

        void unlock() {
            mutex.unlock();
        }
    };
};