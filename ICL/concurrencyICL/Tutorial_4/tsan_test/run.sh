#!/bin/bash
for opt in -O0 -O1 -O2 -O3
do
    echo "Optimization level: $opt"
    clang++ $opt -fsanitize=thread tsan_test.cpp -lpthread
    for i in `seq 1 10`
    do
        ./a.out
    done
done