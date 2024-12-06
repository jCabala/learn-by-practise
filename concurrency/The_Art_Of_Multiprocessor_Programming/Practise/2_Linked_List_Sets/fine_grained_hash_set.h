#pragma once
#include <mutex>
#include <limits>
#include "linked_hash_set.h"

template <class T>
class FineListSet : public Set<T> {
private:
    typename Set<T>::Node* head_;
    std::hash<T> hash_fn_;

public:
    FineListSet(std::hash<T> hash_fn) : hash_fn_(hash_fn) {
        head_ = new typename Set<T>::Node(std::numeric_limits<int>::min());
        head_->next = new typename Set<T>::Node(std::numeric_limits<int>::max());
    }

    ~FineListSet() {
        auto* curr = head_;
        while (curr != nullptr) {
            auto* next = curr->next;
            delete curr;
            curr = next;
        }
    }

    bool add(T x) override {
        int key = hash_fn_(x);
        head_->lock();
        auto* pred = head_;
        auto* curr = pred->next;
        curr->lock();
        while (curr->key < key) {
            pred->unlock();
            pred = curr;
            curr = curr->next;
            curr->lock();
        }

        if (curr->key == key) {
            pred->unlock();
            curr->unlock();
            return false;
        } else {
            auto* newNode = new typename Set<T>::Node(key, x);
            newNode->next = curr;
            pred->next = newNode;
            pred->unlock();
            curr->unlock();
            return true;
        }
    }

    bool remove(T x) override {
        int key = hash_fn_(x);
        head_->lock();
        auto* pred = head_;
        auto* curr = pred->next;
        curr->lock();
        while (curr->key < key) {
            pred->unlock();
            pred = curr;
            curr = curr->next;
            curr->lock();
        }

        if (curr->key == key) {
            pred->next = curr->next;
            pred->unlock();
            curr->unlock();
            delete curr;
            return true;
        } else {
            pred->unlock();
            curr->unlock();
            return false;
        }
    }

    bool contains(T x) override {
        int key = hash_fn_(x);
        head_->lock();
        auto* pred = head_;
        auto* curr = pred->next;
        curr->lock();
        while (curr->key < key) {
            pred->unlock();
            pred = curr;
            curr = curr->next;
            curr->lock();
        }

        bool result = curr->key == key;
        pred->unlock();
        curr->unlock();
        return result;
    }

    void printName() override {
        std::cout << "FineListSet" << std::endl;
    }
};
