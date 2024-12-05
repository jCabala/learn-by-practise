#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

class Consensus {
private:
    std::atomic<int> leader;
    const int NO_LEADER = -1;

public:
    Consensus() {
        leader.store(NO_LEADER);
    }
    bool electLeader(int threadId) {
        int expected = NO_LEADER;
        return leader.compare_exchange_strong(expected, threadId);
    }

    int getLeader() const {
        return leader.load();
    }
};

void threadBody(Consensus& consensus, int threadId, bool print) {
    if (consensus.electLeader(threadId)) {
        if (print) {
            std::cout << "Thread " << threadId << " became the leader!" << std::endl;
        }    
    } else if (print) {
        std::cout << "Thread " << threadId << " failed to become the leader. Current leader is Thread "
                  << consensus.getLeader() << "." << std::endl;
    }
}

int main_print_leader() {
    const int NUM_THREADS = 5;
    Consensus consensus;
    std::vector<std::thread> threads;
    std::cout << "Initial leader value: " << consensus.getLeader() << std::endl;

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(threadBody, std::ref(consensus), i, true);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Final leader: Thread " << consensus.getLeader() << std::endl;
    return 0;
}

int main_leader_different_than_0() {
    const int NUM_THREADS = 5;
    int iterations = 0;
    int leader = 0;

    while(leader == 0) {
        Consensus consensus;
        std::vector<std::thread> threads;

        for (int i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back(threadBody, std::ref(consensus), i, false);
        }

        for (auto& t : threads) {
            t.join();
        }

        leader = consensus.getLeader();
        ++iterations;
    }

    std::cout << "After " << iterations << " iterations, we got a leader different that Thread 0 which is Thread "
              << leader << std::endl;
    return 0;
}

int main() {
    // return main_print_leader();
    return main_leader_different_than_0();
}