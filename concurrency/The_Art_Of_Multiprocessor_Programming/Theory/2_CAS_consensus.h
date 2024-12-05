#ifndef CAS_CONSENSUS_H
#define CAS_CONSENSUS_H
#include <atomic>

template <typename T>
class Consensus {
private:
    std::atomic<T> leader;
    T* values;
    bool decided = false;
    const int NO_LEADER = -1;

public:
    Consensus(int NUM_THREADS) {
        leader.store(NO_LEADER);
        values = new T[NUM_THREADS];
    }
    bool decide(int tid, T val) {
        if (decided) {
            return false;
        }

        int expected = NO_LEADER;
        values[tid] = val;
        decided = true;
        return leader.compare_exchange_strong(expected, tid);
    }

    T getDecision() const {
        if (!decided) {
            throw std::runtime_error("No decision has been made yet.");
        }
        return values[leader.load()];
    }
};
#endif