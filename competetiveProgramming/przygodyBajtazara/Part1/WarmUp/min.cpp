// https://szkopul.edu.pl/problemset/problem/POAyCWzUB990_g4_MA4GF9Jw/site/?key=statement
#include <bits/stdc++.h>

int n;
bool isPlus[1000001];

int main() {
    std::cin >> n;
    for (int i = 0; i < n-1; i++) {
        char c;
        std::cin >> c;
        isPlus[i] = (c == '+');
    }

    bool reversed = false;
    for (int i = 0; i < n - 1; i++) {
        if (isPlus[i] == reversed) {
            std::cout << "-";
        } else if (!isPlus[i] && reversed) {
            std::cout << ")-";
            reversed = false;
        } else { //isPlus && !reversed
            std::cout << "(-";
            reversed = true;
        }
    }

    if (reversed) {
        std::cout << ")";
    }

    std::cout << "\n";
}