// TODO: Fix and finsih

/*#include <string>
#include <iostream>
#include <vector>
#include "2_CAS_consensus.h"

class Invocation {
public:
    Invocation(std::string name) : name(name) {}
    std::string getName() {
        return name;
    } 
    private:
    std::string name;
};

class SequentialObject {
public:
    void apply(Invocation invocation) {
        std::cout << invocation.getName() << std::endl;
    }
};

class Node {
public:
    Node(int NUM_THREADS, const Invocation& invoc_) 
        : invoc(invoc_), 
          decideNext(*new Consensus<Node*>(NUM_THREADS)),
          next(nullptr), 
          seq(0) {}

    ~Node() { 
        delete &decideNext;
        delete next;
    }

    static Node max(std::vector<Node> nodes) {
        int maxIdx = 0;
        for (int i = 1; i < nodes.size(); ++i) {
            if (nodes[i].seq > nodes[maxIdx].seq) {
                maxIdx = i;
            }
        }
        return nodes[maxIdx];
    }

    Invocation invoc;
    Consensus<Node*>& decideNext;
    Node *next;
    int seq;
};

class LockFreeUniversalConstruction {
public:
    LockFreeUniversalConstruction(int NUM_THREADS) 
        : tail(new Node(NUM_THREADS, *new Invocation("tail"))),
          NUM_THREADS(NUM_THREADS) {
        tail->seq = 1;
        for (int i = 0; i < NUM_THREADS; ++i) {
           heads.push_back(tail);
        }
    }

    ~LockFreeUniversalConstruction() {
        delete &tail;
    }

    void apply(Invocation invoc, int tid) {
        Node *prefer = new Node(0, invoc);
        while (prefer->seq == 0) {
            Node *before = heads[0];
            before->decideNext.decide(tid, prefer);
            Node *after = before->decideNext.getDecision();
            before->next = after;
            after->seq = before->seq + 1;
        }
        //...
    }
private:
    std::vector<Node*> heads;
    Node* tail;
    int NUM_THREADS;
};
*/
