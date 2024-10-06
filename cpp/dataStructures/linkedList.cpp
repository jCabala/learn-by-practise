#include <cassert>
#include <iostream>
#include <memory>

template <typename T>
class LinkedList {
public:

    LinkedList() : head_(nullptr), tail_(nullptr) {}

    void Add(T elem) {
        if (head_ == nullptr) {
            assert(tail_ == nullptr && "tail_ should be null iff head_ is null");
            head_ = std::make_unique<Node>(elem);
            tail_ = head_.get();
        } else {
            tail_->next = std::make_unique<Node>(elem);
            tail_ = tail_->next.get();
        }
    }

    bool Contains(T candidate_elem) {
        for (Node* temp = head_.get(); temp != nullptr; temp = temp->next.get()) {
            if (candidate_elem == temp->data) {
                return true;
            }
        }
        return false;
    }

private:
    struct Node {
        Node(T data): data(data), next(nullptr) {}
        T data;
        std::unique_ptr<Node> next;
    };

    std::unique_ptr<Node> head_;
    Node* tail_; // Already owned by a previous element or head
};

int main() {
    std::cout << "Starting..." << std::endl;
    LinkedList<int> my_list;
    my_list.Add(1);
    my_list.Add(2);
    assert(my_list.Contains(2));
    assert(!my_list.Contains(3));
    std::cout << "Assertions passed!" << std::endl;
    return 0;
}