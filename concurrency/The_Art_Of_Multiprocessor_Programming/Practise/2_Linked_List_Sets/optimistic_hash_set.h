#pragma once
#include <mutex>
#include <limits>
#include "linked_hash_set.h"

template <class T>
class OptimisticListSet : public Set<T> {
private:
    typename Set<T>::Node* head_;
    std::hash<T> hash_fn_;
    bool validate(typename Set<T>::Node* pred, typename Set<T>::Node* curr, int key) {
        auto* node = head_;
        while (node->key <= key) {
            if (node == pred) {
                return pred->next == curr;
            }
            node = node->next;
        }
        return false;
    }

public:
    OptimisticListSet(std::hash<T> hash_fn) : hash_fn_(hash_fn) {
        head_ = new typename Set<T>::Node(std::numeric_limits<int>::min());
        head_->next = new typename Set<T>::Node(std::numeric_limits<int>::max());
    }

    ~OptimisticListSet() {
        auto* curr = head_;
        while (curr != nullptr) {
            auto* next = curr->next;
            delete curr;
            curr = next;
        }
    }

    bool add(T x) override {
        int key = hash_fn_(x);
        while(true) {
            auto pred = head_;
            auto curr = pred->next;
            while (curr->key < key) {
                pred = curr;
                curr = curr->next;
            }

            std::scoped_lock lock(pred->mutex, curr->mutex);
            if (validate(pred, curr, key)) {
                if (curr->key == key) {
                    pred->unlock();
                    curr->unlock();
                    return false;
                } else {
                    auto* newNode = new typename Set<T>::Node(key, x);
                    newNode->next = curr;
                    pred->next = newNode;
                    return true;
                }
            }
        }
    }

    bool remove(T x) override {
        int key = hash_fn_(x);
        while(true) {
            auto pred = head_;
            auto curr = pred->next;
            while (curr->key < key) {
                pred = curr;
                curr = curr->next;
            }

            std::scoped_lock lock(pred->mutex, curr->mutex);
            if (validate(pred, curr, key)) {
                if (curr->key == key) {
                    pred->next = curr->next;
                    return true;
                } else {
                    return false;
                }
            }
        }
    }

    bool contains(T x) override {
        int key = hash_fn_(x);
        while(true) {
            auto pred = head_;
            auto curr = pred->next;
            while (curr->key < key) {
                pred = curr;
                curr = curr->next;
            }

            std::scoped_lock lock(pred->mutex, curr->mutex);
            if (validate(pred, curr, key)) {
                return curr->key == key;
            }
        }
    }
    
    void printName() override {
        std::cout << "OptimisticListSet" << std::endl;
    }
};
