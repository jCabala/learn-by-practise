#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include "coarse_grained_hash_set.h"
#include "fine_grained_hash_set.h"
#include "optimistic_hash_set.h"

template <typename TSet>
int run_main(TSet& set) {
    std::atomic<int> success_counter{0};
    const int num_threads = 4;
    const int num_elements_per_thread = 10;

    set.printName();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            int base_value = 100 * i;
            for (int k = 0; k < num_elements_per_thread; ++k) {
                int value = base_value + k * 10;

                // Add
                if (set.add(value)) {
                    success_counter.fetch_add(1);
                } else {
                    std::cerr << "Thread " << i << ": Failed to add " << value << std::endl;
                }

                // Contains
                if (set.contains(value)) {
                    success_counter.fetch_add(1);
                } else {
                    std::cerr << "Thread " << i << ": Failed to find " << value << std::endl;
                }

                // Remove
                if (set.remove(value)) {
                    success_counter.fetch_add(1);
                } else {
                    std::cerr << "Thread " << i << ": Failed to remove " << value << std::endl;
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cout << "Test completed. All threads finished." << std::endl;
    std::cout << "Expected successes: " << num_threads * num_elements_per_thread * 3 << std::endl;
    std::cout << "Successes: " << success_counter.load() << std::endl;

    return 0;
}

int course_list_set_main() {
    CoarseListSet<int> set(std::hash<int>{});
    return run_main(set);
}

int fine_list_set_main() {
    FineListSet<int> set(std::hash<int>{});
    return run_main(set);
}

int optimistic_list_set_main() {
    OptimisticListSet<int> set(std::hash<int>{});
    return run_main(set);
}

int main() {
    //return course_list_set_main();
    // return fine_list_set_main();
    return optimistic_list_set_main();
}
