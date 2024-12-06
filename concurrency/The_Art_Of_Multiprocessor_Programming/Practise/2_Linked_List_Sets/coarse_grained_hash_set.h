#pragma once
#include <mutex>
#include <limits>
#include "linked_hash_set.h"
// ____________________ Coarse Grained ____________________
template <class T>
class CoarseListSet : public Set<T> {
private:
    typename Set<T>::Node* head_;
    std::mutex mutex_;
    std::hash<T> hash_fn_;

public:
    CoarseListSet(std::hash<T> hash_fn) : hash_fn_(hash_fn) {
        head_ = new typename Set<T>::Node(std::numeric_limits<int>::min());
        head_->next = new typename Set<T>::Node(std::numeric_limits<int>::max());
    }

    ~CoarseListSet() {
        auto* curr = head_;
        while (curr != nullptr) {
            auto* next = curr->next;
            delete curr;
            curr = next;
        }
    }

    bool add(T x) override {
        int key = hash_fn_(x);
        std::scoped_lock lock(mutex_);

        auto* pred = head_;
        auto* curr = pred->next;
        while (curr->key < key) {
            pred = curr;
            curr = curr->next;
        }

        if (curr->key == key) {
            return false;
        } else {
            auto* newNode = new typename Set<T>::Node(key, x);
            newNode->next = curr;
            pred->next = newNode;
            return true;
        }
    }

    bool remove(T x) override {
        int key = hash_fn_(x);
        std::scoped_lock lock(mutex_);

        auto* pred = head_;
        auto* curr = pred->next;
        while (curr->key < key) {
            pred = curr;
            curr = curr->next;
        }

        if (curr->key == key) {
            pred->next = curr->next;
            delete curr;
            return true;
        } else {
            return false;
        }
    }

    bool contains(T x) override {
        int key = hash_fn_(x);
        std::scoped_lock lock(mutex_);

        auto* pred = head_;
        auto* curr = pred->next;
        while (curr->key < key) {
            pred = curr;
            curr = curr->next;
        }

        return curr->key == key;
    }

    void printName() override {
        std::cout << "CoarseListSet" << std::endl;
    }
};